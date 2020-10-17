#!/bin/bash

l=$(wc -l $1 | tr -dc '0-9')
n=$(($l-1))
n1=$(($n*$2/100))
n2=$(($n*$3/100))
n3=$(($n-($n1+$n2)))

d1=$(($n1+1))
d2=$(($d1+$n2-1))
d3=$(($d2+$n3-1))

head -n 1 $1 > `dirname $1`/`basename $1 .csv`_1.csv
sed -n "2,${d1}p" $1 >> `dirname $1`/`basename $1 .csv`_1.csv

head -n 1 $1 > `dirname $1`/`basename $1 .csv`_2.csv
sed -n "${d1},${d2}p" $1 >> `dirname $1`/`basename $1 .csv`_2.csv

head -n 1 $1 > `dirname $1`/`basename $1 .csv`_3.csv
sed -n "${d2},${d3}p" $1 >> `dirname $1`/`basename $1 .csv`_3.csv

## eof - $RCSfile$
