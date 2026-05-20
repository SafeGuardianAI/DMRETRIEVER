import os

TASK2PREFIX = {
    "FactCheck": "Given the claim, retrieve most relevant document that supports or refutes the claim",
    "NLI":       "Given the premise, retrieve most relevant hypothesis that is entailed by the premise",
    "QA":        "Given the question, retrieve most relevant passage that best answers the question",
    "QAdoc":     "Given the question, retrieve the most relevant document that answers the question",
    "STS":       "Given the sentence, retrieve the sentence with the same meaning",
    "Twitter":   "Given the user query, retrieve the most relevant Twitter text that meets the request",
}

_DATA_ROOT = os.environ.get("DMRETRIEVER_DATA_ROOT", "data")

CORPUS_DIR         = os.environ.get("DMRETRIEVER_CORPUS_DIR",         f"{_DATA_ROOT}/C_test_set/corpus")
TEST_QUERY_DIR     = os.environ.get("DMRETRIEVER_TEST_QUERY_DIR",     f"{_DATA_ROOT}/C_test_set/test_query_eva")
BASELINE_INDEX_DIR = os.environ.get("DMRETRIEVER_BASELINE_INDEX_DIR", f"{_DATA_ROOT}/E_test_res/corpus_embeddings")
LABEL_POOL_DIR     = os.environ.get("DMRETRIEVER_LABEL_POOL_DIR",     f"{_DATA_ROOT}/E_test_res/label_pools")
QUERY_EMB_DIR      = os.environ.get("DMRETRIEVER_QUERY_EMB_DIR",      f"{_DATA_ROOT}/E_test_res/query_embeddings")
CHECKPOINT_ROOT    = os.environ.get("DMRETRIEVER_CHECKPOINT_ROOT",     f"{_DATA_ROOT}/D_train_output")
PERF_DIR           = os.environ.get("DMRETRIEVER_PERF_DIR",           f"{_DATA_ROOT}/E_test_res/performance")
QRELS_DIR          = os.environ.get("DMRETRIEVER_QRELS_DIR",          f"{_DATA_ROOT}/C_test_set/qrels_with_added")
RAW_DATA_DIR       = os.environ.get("DMRETRIEVER_RAW_DATA_DIR",       f"{_DATA_ROOT}/A_A_reb")

DEFAULT_BATCH  = 4
DEFAULT_MAXLEN = 512
DEFAULT_TOPK   = 10
USE_FP16       = True
