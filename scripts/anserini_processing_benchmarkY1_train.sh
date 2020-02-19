
wget http://trec-car.cs.unh.edu/datareleases/v2.0/benchmarkY1-train.v2.0.tar.xz
tar -xf benchmarkY1-train.v2.0.tar.xz

THIS_DIR=./

cp ${THIS_DIR}/benchmarkY1/benchmarkY1-train/fold-0-train.pages.cbor-hierarchical.qrels ${THIS_DIR}/train_temp.qrels
cp ${THIS_DIR}/benchmarkY1/benchmarkY1-train/fold-1-train.pages.cbor-hierarchical.qrels ${THIS_DIR}/dev_temp.qrels

grep -v -e "%[^0-9A-F]" -e "%[0-9A-F][^0-9A-F]" ${THIS_DIR}/train_temp.qrels > ${THIS_DIR}/toy_train.qrels
grep -v -e "%[^0-9A-F]" -e "%[0-9A-F][^0-9A-F]" ${THIS_DIR}/dev_temp.qrels > ${THIS_DIR}/toy_dev.qrels

rm -f ${THIS_DIR}/train_temp.qrels
rm -f ${THIS_DIR}/dev_temp.qrels

sort -u -o ${THIS_DIR}/toy_train.qrels ${THIS_DIR}/toy_train.qrels
sort -u -o ${THIS_DIR}/toy_dev.qrels ${THIS_DIR}/toy_dev.qrels

cat ${THIS_DIR}/toy_train.qrels | cut -d' ' -f1 > ${THIS_DIR}/toy_train.topics
cat ${THIS_DIR}/toy_dev.qrels | cut -d' ' -f1 > ${THIS_DIR}/toy_dev.topics

sort -u -o ${THIS_DIR}/toy_train.topics ${THIS_DIR}/toy_train.topics
sort -u -o ${THIS_DIR}/toy_dev.topics ${THIS_DIR}/toy_dev.topics

