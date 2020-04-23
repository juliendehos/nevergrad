#!/bin/bash

# loss dimension budget mu lambda

filename="data.csv"
rm -f tmp tmp2

sed '1d' $1 > tmp
cut -d, -f2-20,23-24 --complement tmp > tmp2
awk 'BEGIN{FS=OFS=","} {gsub(/\_/, ",", $4)} 1' tmp2 > tmp
cut -d, -f4 --complement tmp > ${filename}
sed -i '1 i\loss,dimension,budget,mu,lambda' ${filename}


# python cleanandlearn.py ${filename}
