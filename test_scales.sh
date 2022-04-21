rows=`cat scale_list`

for row in ${rows[@]}
do
  echo "test $row"
  ./test_scale.sh output/raw/$row
done