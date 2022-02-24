import argparse
import csv
import json
import logging
import multiprocessing
import os
import tempfile
import time
import unicodedata
from collections import defaultdict
from contextlib import closing
from typing import List

import torch
import faiss
import joblib
import numpy as np
import regex
from torch.nn.parallel import DataParallel
from tqdm import tqdm
import statistics
import bpr.index
from bpr.biencoder import BiEncoder
from bpr.index import FaissBinaryIndex, FaissIndex, FaissHNSWIndex
from bpr.passage_db import PassageDB
from bpr.retriever import Retriever
from transformers import DPRReader, DPRReaderTokenizer

logger = logging.getLogger(__name__)

# https://github.com/facebookresearch/DPR/blob/f403c3b3e179e53c0fe68a0718d5dc25371fe5df/dpr/utils/tokenizers.py#L154
ALPHA_NUM = r"[\p{L}\p{N}\p{M}]+"
NON_WS = r"[^\p{Z}\p{C}]"
# https://github.com/facebookresearch/DPR/blob/f403c3b3e179e53c0fe68a0718d5dc25371fe5df/dpr/utils/tokenizers.py#L163
REGEXP = regex.compile("(%s)|(%s)" % (ALPHA_NUM, NON_WS), flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE)

class Timer:
    def __init__(self):
        self.starter = torch.cuda.Event(enable_timing=True)
        self.ender = torch.cuda.Event(enable_timing=True)
    def start(self):
        self.starter.record()
    def end(self):
        self.ender.record()
        torch.cuda.synchronize()
        elapsed = self.starter.elapsed_time(self.ender)
        return elapsed

def _has_answer(answer: str, passage: str) -> bool:
    def tokenize(text: str) -> List[str]:
        return [m.group() for m in REGEXP.finditer(text)]

    answer_tokens = tokenize(unicodedata.normalize("NFD", answer.lower()))
    passage_tokens = tokenize(unicodedata.normalize("NFD", passage.lower()))

    for i in range(0, len(passage_tokens) - len(answer_tokens) + 1):
        if answer_tokens == passage_tokens[i : i + len(answer_tokens)]:  # noqa: E203
            return True
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--biencoder_file", type=str, required=True)
    parser.add_argument("--embedding_file", type=str, required=True)
    parser.add_argument("--passage_db_file", type=str, required=True)
    parser.add_argument("--qa_file", type=str, required=True)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--top_k", type=int, default=[1, 5, 20, 50, 100], action="append")
    parser.add_argument("--no_eval", action="store_true")
    parser.add_argument("--binary_k", type=int, default=2048)
    parser.add_argument("--binary_to_float", action="store_true")
    parser.add_argument("--binary_no_rerank", action="store_true")
    parser.add_argument("--use_binary_hash", action="store_true")
    parser.add_argument("--use_hnsw", action="store_true")
    parser.add_argument("--hash_num_bits", type=int, default=768)
    parser.add_argument("--hnsw_store_n", type=int, default=512)
    parser.add_argument("--hnsw_ef_construction", type=int, default=200)
    parser.add_argument("--hnsw_ef_search", type=int, default=128)
    parser.add_argument("--biencoder_device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--index_device", type=str, default="cpu", choices=["cuda", "cpu"])
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--pool_size", type=int, default=multiprocessing.cpu_count())
    parser.add_argument("--chunk_size", type=int, default=32)
    parser.add_argument("--retriever_batch_size", type=int, default=32)
    parser.add_argument("--reader_batch_size", type=int, default=100)

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.WARNING, format="[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)"
    )
    logger.setLevel(logging.INFO)
    bpr.index.logger.setLevel(logging.INFO)

    def process_candidates(args):
        (query, answers), candidates = args
        passage_dicts = []
        for index, candidate in enumerate(candidates):
            passage_dict = dict(id=int(candidate.passage.id), score=float(candidate.score))
            passage_dict["title"] = candidate.passage.title
            passage_dict["text"] = candidate.passage.text

            if any(_has_answer(answer, candidate.passage.text) for answer in answers):
                passage_dict["has_answer"] = True
            else:
                passage_dict["has_answer"] = False

            passage_dicts.append(passage_dict)

        return dict(question=query, answers=answers, ctxs=passage_dicts)

    tokenizer = DPRReaderTokenizer.from_pretrained('facebook/dpr-reader-single-nq-base')
    model = DPRReader.from_pretrained('facebook/dpr-reader-single-nq-base').to(torch.device("cuda"))

    passage_db = PassageDB(args.passage_db_file)
    embedding_data = joblib.load(args.embedding_file, mmap_mode="r")
    ids, embeddings = embedding_data["ids"], embedding_data["embeddings"]
    dim_size = embeddings.shape[1]

    logger.info("Building index...")
    if embeddings.dtype == np.uint8:
        if args.binary_to_float:
            embeddings = np.unpackbits(embeddings).reshape(-1, dim_size * 8).astype(np.float32)
            embeddings = embeddings * 2 - 1
            base_index = faiss.IndexFlatIP(dim_size * 8)
            index = FaissIndex.build(ids, embeddings, base_index)

        elif args.use_binary_hash:
            base_index = faiss.IndexBinaryHash(dim_size * 8, args.hash_num_bits)
            index = FaissBinaryIndex.build(ids, embeddings, base_index)

        else:
            #-------
            if torch.cuda.is_available():
                base_index = faiss.read_index_binary("./training_files/nq-gpu-full.idx")
                res = faiss.StandardGpuResources()
                index = faiss.GpuIndexBinaryFlat(res, base_index)
                index = FaissBinaryIndex(index,ids,embeddings)
            else:
                base_index = faiss.IndexBinaryFlat(dim_size * 8)
                index = FaissBinaryIndex.build(ids, embeddings, base_index)

    elif args.use_hnsw:
        base_index = faiss.IndexHNSWFlat(dim_size + 1, args.hnsw_store_n)
        base_index.hnsw.efSearch = args.hnsw_ef_search
        base_index.hnsw.efConstruction = args.hnsw_ef_construction
        index = FaissHNSWIndex.build(ids, embeddings, base_index)

    else:
        base_index = faiss.IndexFlatIP(dim_size)
        index = FaissIndex.build(ids, embeddings, base_index)
        if args.index_device == "cuda":
            index = index.to_gpu()

    del ids
    del embeddings

    logger.info("Loading BiEncoder...")
    biencoder = BiEncoder.load_from_checkpoint(args.biencoder_file, map_location="cpu",strict=False)
    biencoder = biencoder.to(args.biencoder_device)
    biencoder.eval()
    biencoder.freeze()

    if args.parallel:
        biencoder.query_encoder = DataParallel(biencoder.query_encoder)

    retriever = Retriever(index, biencoder, passage_db)

    logger.info("Loading QA pairs from %s", args.qa_file)
    with open(args.qa_file) as f:
        qa_pairs = [(row[0], eval(row[1].strip())) for row in csv.reader(f, delimiter="\t")]
    qa_pairs = qa_pairs[:args.chunk_size]
    total_count = len(qa_pairs)



    logger.info("Computing query embeddings...")
    queries = [pair[0] for pair in qa_pairs]
    encoding_timer = Timer()
    encoding_timer.start()
    query_embeddings = retriever.encode_queries(queries)
    logger.info("complete! time took:%f per query, %d queries in total",encoding_timer.end()/total_count,total_count)

    def iterator(batch_size, *args):
        for i in range(0, len(args[0])-batch_size+1, batch_size):
            yield (arg[i:i + batch_size] for arg in args)

    search_latencies = []
    preprocess_latencies = []
    answer_extract_latencies = []
    postprocess_latencies = []
    total_latencies = []
    retriever_memories = []
    reader_memories = []

    #------------timers---------------
    search_timer = Timer()
    preprocess_timer = Timer()
    answer_extracting_timer = Timer()
    postprocess_timer = Timer()
    encoding_timer = Timer()
    total_latency_timer = Timer()
    MB = (1024.0 * 1024.0)

    logger.info("Getting top-k results...")


    start_time = time.time()
    for query_embedding, questiona in iterator(args.retriever_batch_size, query_embeddings, qa_pairs):
        torch.cuda.reset_max_memory_allocated()
        start = torch.cuda.max_memory_allocated() / MB
        print(start)
        total_latency_timer.start()

        search_timer.start()
        print(query_embedding.shape)
        if isinstance(index, FaissBinaryIndex):
            topk_results = retriever.search(
                query_embedding, max(args.top_k), binary_k=args.binary_k, rerank=not args.binary_no_rerank
            )
        else:
            topk_results = retriever.search(query_embedding, max(args.top_k))
        query_time = search_timer.end()
        search_latencies.append(query_time)
        logger.info("Elapsed time: %.2fsec", query_time)
        print(torch.cuda.max_memory_allocated()/MB)
        size = (torch.cuda.max_memory_allocated()/MB) - start
        retriever_memories.append(size)
        print(size)


        torch.cuda.reset_max_memory_allocated()
        start = torch.cuda.max_memory_allocated() / MB
        texts = []
        reader_questions = []
        titles = []
        ground_truth_answers = []
        for result,qa in zip(topk_results,questiona):
            for x in result:
                reader_questions.append(qa[0])
                titles.append(x.passage.title)
                texts.append(x.passage.text)
                ground_truth_answers.append(qa[1])
        for reader_questions,titles,texts,ground_truth_answers in iterator(args.reader_batch_size,reader_questions,titles,texts,ground_truth_answers):
            answers = []
            preprocess_timer.start()
            input_id = tokenizer(questions=reader_questions,
                                 titles=titles,
                                 texts=texts,
                                 return_tensors='pt',padding=True)["input_ids"]
            x = preprocess_timer.end()
            preprocess_latencies.append(x)
            #------------------------------------answer extraction-----------------------------------------------------
            answer_extracting_timer.start()
            input_id = torch.tensor(input_id).to(torch.device("cuda"))
            print(len(input_id))
            with torch.no_grad():
                outputs = model(input_id)
                start_logits = outputs.start_logits
                end_logits = outputs.end_logits

            x = answer_extracting_timer.end()
            answer_extract_latencies.append(x)

            postprocess_timer.start()
            #print out the result
            answer_start = torch.argmax(start_logits, axis = 1)
            answer_end = torch.argmax(end_logits, axis = 1)
            # Combine the tokens in the answer and print it out.
            for i in range(input_id.shape[0]):
                answers.append(' '.join(tokenizer.convert_ids_to_tokens(list(input_id[i][answer_start[i]:answer_end[i]+1]))))
            x = postprocess_timer.end()
            postprocess_latencies.append(x)


        total_latency_time = total_latency_timer.end()
        total_latencies.append(total_latency_time)

        size = (torch.cuda.max_memory_allocated() / MB) - start
        reader_memories.append(size)
        ground_truth_answers = [item for sublist in ground_truth_answers for item in sublist]
    logger.info("------------------------------latency statistics-------------------------")
    logger.info("average search latency: %.2f ms",statistics.mean(search_latencies[:]))
    logger.info("average preprocess latency: %.2f ms",statistics.mean(preprocess_latencies[:]))
    logger.info("average answer extraction latency: %.2f ms",statistics.mean(answer_extract_latencies[:]))
    logger.info("average postprocess latency: %.2f ms",statistics.mean(postprocess_latencies[:]))
    logger.info("average total latency: %.2f ms",statistics.mean(total_latencies[:]))
    logger.info("-------------------------runtime memory---------------------------------")
    logger.info("average retriever memories: %.2f ",statistics.mean(retriever_memories[:]))
    logger.info("average reader memories: %.2f ",statistics.mean(reader_memories[:]))
