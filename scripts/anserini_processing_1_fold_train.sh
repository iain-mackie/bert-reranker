
wget http://trec-car.cs.unh.edu/datareleases/v2.0/train.v2.0.tar.xz
tar -xf train.v2.0.tar.xz

THIS_DIR=./

for f in ${THIS_DIR}/train/fold-[0]-base.train.cbor-hierarchical.qrels; do (cat "${f}"; echo); done > ${THIS_DIR}/train_temp.qrels

grep -v -e "%[^0-9A-F]" -e "%[0-9A-F][^0-9A-F]" ${THIS_DIR}/train_temp.qrels > ${THIS_DIR}/train_fold_0_train_hierarchical.qrels

rm -f ${THIS_DIR}/train_temp.qrels

sort -u -o ${THIS_DIR}/train_fold_0_train_hierarchical.qrels ${THIS_DIR}/train_fold_0_train_hierarchical.qrels

cat ${THIS_DIR}/train_fold_0_train_hierarchical.qrels | cut -d' ' -f1 > ${THIS_DIR}/train_fold_0_train_hierarchical.topics

sort -u -o ${THIS_DIR}/train_fold_0_train_hierarchical.topics ${THIS_DIR}/train_fold_0_train_hierarchical.topics

