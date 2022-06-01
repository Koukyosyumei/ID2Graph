while getopts o: OPT
do
  case $OPT in
    "o" ) FLG_O="TRUE" ; VALUE_O="$OPTARG" ;;
  esac
done

cat "${VALUE_O}/result.ans" | grep -oP '(?<=Tree-1: )(.*)' > "${VALUE_O}/temp_tree_1.out"
cat "${VALUE_O}/result.ans" | grep -oP '(?<=Tree-2: )(.*)' > "${VALUE_O}/temp_tree_2.out"