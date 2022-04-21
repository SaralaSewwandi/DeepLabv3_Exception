#!/bin/bash

output_file='index.html'

cd output

list=`ls -l raw/|awk '{print $9}'|sort -r`

echo '<html><body>' > $output_file

for row in ${list[@]};
do
  # title
  echo "<div>$row</div>" >> $output_file
  # picture
  echo '<div>' >> $output_file
  echo "<img src=\"raw/$row\" />" >> $output_file
  echo "<img src=\"gt/$row\" />" >> $output_file
  echo "<img src=\"mask/$row\" />" >> $output_file
  echo '</div>' >> $output_file
done

echo '</body></html>' >> $output_file
