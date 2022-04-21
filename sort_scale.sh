#!/bin/bash

output_file='scale.html'

scales=(0.8 1 1.2 1.4 1.6 1.8 2 2.2 2.4)

cd output


# 获取文件清单
list=`ls -l scale_0.8/|awk '{print $9}'|sort -r`

echo '<html><body>' > $output_file

for row in ${list[@]};
do
  # title
  echo "<div>$row</div>" >> $output_file
  # picture
  echo '<div>' >> $output_file

  # 插入各个尺度
  for scale in ${scales[@]}
  do
    echo "<img src=\"scale_$scale/$row\" />" >> $output_file
  done

  echo '</div>' >> $output_file
done

echo '</body></html>' >> $output_file
