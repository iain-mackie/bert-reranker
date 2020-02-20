
# wget http://trec-car.cs.unh.edu/datareleases/v2.0/benchmarkY1-train.v2.0.tar.xz
# tar -xf benchmarkY1-train.v2.0.tar.xz

THIS_DIR=./

for f in ${THIS_DIR}/benchmarkY1/benchmarkY1-train/fold-[0-3]-train.pages.cbor-hierarchical.qrels; do (cat "${f}"; echo); done > ${THIS_DIR}/train_temp.qrels
for f in ${THIS_DIR}/benchmarkY1/benchmarkY1-train/fold-[4-5]-train.pages.cbor-hierarchical.qrels; do (cat "${f}"; echo); done > ${THIS_DIR}/dev_temp.qrels

grep -v -e "%[^0-9A-F]" -e "%[0-9A-F][^0-9A-F]" ${THIS_DIR}/train_temp.qrels > ${THIS_DIR}/toy_train_large.qrels
grep -v -e "%[^0-9A-F]" -e "%[0-9A-F][^0-9A-F]" ${THIS_DIR}/dev_temp.qrels > ${THIS_DIR}/toy_dev_large.qrels

rm -f ${THIS_DIR}/train_temp.qrels
rm -f ${THIS_DIR}/dev_temp.qrels

sort -u -o ${THIS_DIR}/toy_train_large.qrels ${THIS_DIR}/toy_train_large.qrels
sort -u -o ${THIS_DIR}/toy_dev_large.qrels ${THIS_DIR}/toy_dev_large.qrels

cat ${THIS_DIR}/toy_train_large.qrels | cut -d' ' -f1 > ${THIS_DIR}/toy_train_large.topics
cat ${THIS_DIR}/toy_dev_large.qrels | cut -d' ' -f1 > ${THIS_DIR}/toy_dev_large.topics

sort -u -o ${THIS_DIR}/toy_train_large.topics ${THIS_DIR}/toy_train_large.topics
sort -u -o ${THIS_DIR}/toy_dev_large.topics ${THIS_DIR}/toy_dev_large.topics

