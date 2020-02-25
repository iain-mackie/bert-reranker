
# wget http://trec-car.cs.unh.edu/datareleases/v2.0/benchmarkY1-train.v2.0.tar.xz
# tar -xf benchmarkY1-train.v2.0.tar.xz

THIS_DIR=./

for f in ${THIS_DIR}/benchmarkY1/benchmarkY1-train/fold-[1-5]-train.pages.cbor-hierarchical.qrels; do (cat "${f}"; echo); done > ${THIS_DIR}/train_temp.qrels
for f in ${THIS_DIR}/benchmarkY1/benchmarkY1-train/fold-[0]-train.pages.cbor-hierarchical.qrels; do (cat "${f}"; echo); done > ${THIS_DIR}/dev_temp.qrels

grep -v -e "%[^0-9A-F]" -e "%[0-9A-F][^0-9A-F]" ${THIS_DIR}/train_temp.qrels > ${THIS_DIR}/train_benchmarkY1.qrels
grep -v -e "%[^0-9A-F]" -e "%[0-9A-F][^0-9A-F]" ${THIS_DIR}/dev_temp.qrels > ${THIS_DIR}/dev_benchmarkY1.qrels

rm -f ${THIS_DIR}/train_temp.qrels
rm -f ${THIS_DIR}/dev_temp.qrels

sort -u -o ${THIS_DIR}/train_benchmarkY1.qrels ${THIS_DIR}/train_benchmarkY1.qrels
sort -u -o ${THIS_DIR}/dev_benchmarkY1.qrels ${THIS_DIR}/dev_benchmarkY1.qrels

cat ${THIS_DIR}/train_benchmarkY1.qrels | cut -d' ' -f1 > ${THIS_DIR}/train_benchmarkY1.topics
cat ${THIS_DIR}/dev_benchmarkY1.qrels | cut -d' ' -f1 > ${THIS_DIR}/dev_benchmarkY1.topics

sort -u -o ${THIS_DIR}/train_benchmarkY1.topics ${THIS_DIR}/train_benchmarkY1.topics
sort -u -o ${THIS_DIR}/dev_benchmarkY1.topics ${THIS_DIR}/dev_benchmarkY1.topics

