cat result.ans | grep -oP '(?<=Tree-1: )(.*)' > temp_tree_1.out
cat result.ans | grep -oP '(?<=Tree-2: )(.*)' > temp_tree_2.out