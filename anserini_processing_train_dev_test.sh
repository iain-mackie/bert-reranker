TRECCAR_DIR=/home/imackie/Documents/trec_car

THIS_DIR=./

for f in ${TRECCAR_DIR}/data/train/fold-[0-3]-base.train.cbor-hierarchical.qrels; do (cat "${f}"; echo); done > ${THIS_DIR}/train_temp.qrels
cp ${TRECCAR_DIR}/data/train/fold-4-base.train.cbor-hierarchical.qrels ${THIS_DIR}/dev_temp.qrels
cp ${TRECCAR_DIR}/data/benchmarkY1/test.pages.cbor-hierarchical.qrels ${THIS_DIR}/test_temp.qrels

grep -v -e "%[^0-9A-F]" -e "%[0-9A-F][^0-9A-F]" ${THIS_DIR}/train_temp.qrels > ${THIS_DIR}/train.qrels
grep -v -e "%[^0-9A-F]" -e "%[0-9A-F][^0-9A-F]" ${THIS_DIR}/dev_temp.qrels > ${THIS_DIR}/dev.qrels
grep -v -e "%[^0-9A-F]" -e "%[0-9A-F][^0-9A-F]" ${THIS_DIR}/test_temp.qrels > ${THIS_DIR}/test.qrels

rm -f ${THIS_DIR}/train_temp.qrels
rm -f ${THIS_DIR}/dev_temp.qrels
rm -f ${THIS_DIR}/test_temp.qrels

sort -u -o ${THIS_DIR}/train.qrels ${THIS_DIR}/train.qrels
sort -u -o ${THIS_DIR}/dev.qrels ${THIS_DIR}/dev.qrels
sort -u -o ${THIS_DIR}/test.qrels ${THIS_DIR}/test.qrels

cat ${THIS_DIR}/train.qrels | cut -d' ' -f1 > ${THIS_DIR}/train.topics
cat ${THIS_DIR}/dev.qrels | cut -d' ' -f1 > ${THIS_DIR}/dev.topics
cat ${THIS_DIR}/test.qrels | cut -d' ' -f1 > ${THIS_DIR}/test.topics

sort -u -o ${THIS_DIR}/train.topics ${THIS_DIR}/train.topics
sort -u -o ${THIS_DIR}/dev.topics ${THIS_DIR}/dev.topics
sort -u -o ${THIS_DIR}/test.topics ${THIS_DIR}/test.topics


