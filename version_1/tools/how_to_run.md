## For Retriever only

# Halubench

python tools/evaluate_rag.py \
  --dataset-prefix data/halubench/halubench \
  --mode retriever \
  --topk 5 \
  --retriever dense

# Hotpot

python tools/evaluate_rag.py \
  --dataset-prefix data/hotpot/hotpot \
  --mode retriever \
  --topk 5 \
  --retriever dense


## For Retrieval + Generation 

# Halubench

python tools/evaluate_rag.py \
  --dataset-prefix data/halubench/halubench \
  --mode rag \
  --topk 5 \
  --retriever dense


# To evaluate both retriever and rag 
python tools/evaluate_rag.py \
  --dataset-prefix data/halubench/halubench \
  --mode both \
  --topk 5 \
  --retriever dense \
  --output results/halubench_eval