while getopts d:m:p:n:f:i:r:c:j:e:w OPT
do
  case $OPT in
    "d" ) FLG_D="TRUE" ; VALUE_D="$OPTARG" ;;
    "m" ) FLG_M="TRUE" ; VALUE_M="$OPTARG" ;;
    "p" ) FLG_P="TRUE" ; VALUE_P="$OPTARG" ;;
    "n" ) FLG_N="TRUE" ; VALUE_N="$OPTARG" ;;
    "f" ) FLG_F="TRUE" ; VALUE_F="$OPTARG" ;;
    "i" ) FLG_I="TRUE" ; VALUE_I="$OPTARG" ;;
    "r" ) FLG_R="TRUE" ; VALUE_R="$OPTARG" ;;
    "c" ) FLG_C="TRUE" ; VALUE_C="$OPTARG" ;;
    "j" ) FLG_J="TRUE" ; VALUE_J="$OPTARG" ;;
    "e" ) FLG_E="TRUE" ; VALUE_E="$OPTARG" ;;
    "w" ) FLG_W="TRUE" ; VALUE_W="$OPTARG" ;;
  esac
done

if [ "${VALUE_M}" = "xgboost" ] || [ "${VALUE_M}" = "x" ]; then
  g++ -pthread -o script/build/pipeline_1_training.out script/pipeline_1_train_xgboost.cpp
  g++ -o script/build/pipeline_2_louvain.out script/pipeline_2_louvain.cpp

  for i in $(seq 1 5)
  do 
      echo "random seed is $i"
      python3 ./data/prep.py -d ${VALUE_D} -p "./data/${VALUE_D}/" -n ${VALUE_N} -f ${VALUE_F} -i ${VALUE_I} -s ${i}
      cp "./data/${VALUE_D}/${VALUE_D}.in" "${VALUE_P}/${i}_data.in"
      if [ "${FLG_W}" = "TRUE" ]; then
        script/build/pipeline_1_training.out -f ${VALUE_P} -p ${i} -r ${VALUE_R} -c ${VALUE_C} -j ${VALUE_J} -w < "${VALUE_P}/${i}_data.in"
      else
        script/build/pipeline_1_training.out -f ${VALUE_P} -p ${i} -r ${VALUE_R} -c ${VALUE_C} -j ${VALUE_J} < "${VALUE_P}/${i}_data.in"
      fi
      script/build/pipeline_2_louvain.out -c ${VALUE_C} -e ${VALUE_E}$ < "${VALUE_P}/${i}_adj_mat.txt" > "${VALUE_P}/${i}_communities.out"
  done
elif [ "${VALUE_M}" = "randomforest" ] || [ "${VALUE_M}" = "r" ]; then
  g++ -pthread -o script/build/pipeline_1_training.out script/pipeline_1_train_randomforest.cpp
  g++ -o script/build/pipeline_2_louvain.out script/pipeline_2_louvain.cpp

  for i in $(seq 1 5)
  do 
      echo "random seed is $i"
      python3 ./data/prep.py -d ${VALUE_D} -p "./data/${VALUE_D}/" -n ${VALUE_N} -f ${VALUE_F} -i ${VALUE_I} -s ${i}
      cp "./data/${VALUE_D}/${VALUE_D}.in" "${VALUE_P}/${i}_data.in"
      if [ "${FLG_W}" = "TRUE" ]; then
        script/build/pipeline_1_training.out -f ${VALUE_P} -p ${i} -r ${VALUE_R} -j ${VALUE_J} -w < "${VALUE_P}/${i}_data.in"
      else
        script/build/pipeline_1_training.out -f ${VALUE_P} -p ${i} -r ${VALUE_R} -j ${VALUE_J} < "${VALUE_P}/${i}_data.in"
      fi
      script/build/pipeline_2_louvain.out -c ${VALUE_C} -e ${VALUE_E}$ < "${VALUE_P}/${i}_adj_mat.txt" > "${VALUE_P}/${i}_communities.out"
  done
else
  echo "m=${VALUE_M} is not supported"
fi