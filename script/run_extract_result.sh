while getopts o: OPT
do
  case $OPT in
    "o" ) FLG_O="TRUE" ; VALUE_O="$OPTARG" ;;
  esac
done

cat "${VALUE_O}/result.ans" | grep -oP '(?<=Tree-1: )(.*)' > "${VALUE_O}/temp_lp_tree_1.out"
cat "${VALUE_O}/result.ans" | grep -oP '(?<=Tree-2: )(.*)' > "${VALUE_O}/temp_lp_tree_2.out"
cat "${VALUE_O}/result.ans" | grep -oP '(?<=Train AUC: )(.*)' > "${VALUE_O}/temp_train_auc.out"
cat "${VALUE_O}/result.ans" | grep -oP '(?<=Val AUC: )(.*)' > "${VALUE_O}/temp_val_auc.out"