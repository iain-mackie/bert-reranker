TRECCAR_DIR=/home/imackie/Documents/trec_car

THIS_DIR=./

cp ${TRECCAR_DIR}/data/train/base.train.cbor-hierarchical.qrels ${THIS_DIR}/train_hierarchical_temp.qrels
cp ${TRECCAR_DIR}/data/train_tree/base.train.cbor.tree.qrels ${THIS_DIR}/train_temp.qrels

grep -v -e "%[^0-9A-F]" -e "%[0-9A-F][^0-9A-F]" ${THIS_DIR}/train_hierarchical_temp.qrels > ${THIS_DIR}/train_hierarchical.qrels
grep -v -e "%[^0-9A-F]" -e "%[0-9A-F][^0-9A-F]" ${THIS_DIR}/train_temp.qrels > ${THIS_DIR}/train.qrels

rm -f ${THIS_DIR}/train_hierarchical_temp.qrels
rm -f ${THIS_DIR}/train_temp.qrels

sort -u -o ${THIS_DIR}/train_hierarchical.qrels ${THIS_DIR}/train_hierarchical.qrels
sort -u -o ${THIS_DIR}/train.qrels ${THIS_DIR}/train.qrels

grep '/' ${THIS_DIR}/train.qrels > ${THIS_DIR}/train_no_root.qrels

cat ${THIS_DIR}/train_hierarchical.qrels | cut -d' ' -f1 > ${THIS_DIR}/train_hierarchical.topics
cat ${THIS_DIR}/train.qrels | cut -d' ' -f1 > ${THIS_DIR}/train.topics
cat ${THIS_DIR}/train_no_root.qrels | cut -d' ' -f1 > ${THIS_DIR}/train_no_root.topics

sort -u -o ${THIS_DIR}/train_hierarchical.topics ${THIS_DIR}/train_hierarchical.topics
sort -u -o ${THIS_DIR}/train.topics ${THIS_DIR}/train.topics
sort -u -o ${THIS_DIR}/train_no_root.topics ${THIS_DIR}/train_no_root.topics


