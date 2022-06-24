while getopts d:m:p:n:f:i:r:c:h:j:e:l:z:k:s:w OPT
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
    "h" ) FLG_H="TRUE" ; VALUE_H="$OPTARG" ;;
    "j" ) FLG_J="TRUE" ; VALUE_J="$OPTARG" ;;
    "e" ) FLG_E="TRUE" ; VALUE_E="$OPTARG" ;;
    "l" ) FLG_L="TRUE" ; VALUE_L="$OPTARG" ;;
    "z" ) FLG_Z="TRUE" ; VALUE_Z="$OPTARG" ;;
    "k" ) FLG_K="TRUE" ; VALUE_K="$OPTARG" ;;
    "s" ) FLG_S="TRUE" ; VALUE_S="$OPTARG" ;;
    "w" ) FLG_W="TRUE" ; VALUE_W="$OPTARG" ;;
  esac
done

echo "random seed is ${VALUE_S}"
python3 ./data/prep.py -d ${VALUE_D} -p "./data/${VALUE_D}/" -n ${VALUE_N} -f ${VALUE_F} -i ${VALUE_I} -s ${VALUE_S}
cp "./data/${VALUE_D}/${VALUE_D}_${VALUE_S}.in" "${VALUE_P}/${VALUE_S}_data.in"

if [ "${VALUE_M}" = "xgboost" ] || [ "${VALUE_M}" = "x" ]; then
  if [ "${FLG_W}" = "TRUE" ]; then
    script/build/pipeline_1_training.out -f ${VALUE_P} -p ${VALUE_S} -r ${VALUE_R} -c ${VALUE_C} -e ${VALUE_E}$ -l ${VALUE_L} -z ${VALUE_Z} -h ${VALUE_H} -j ${VALUE_J} -w < "${VALUE_P}/${VALUE_S}_data.in"
  else
    script/build/pipeline_1_training.out -f ${VALUE_P} -p ${VALUE_S} -r ${VALUE_R} -c ${VALUE_C} -e ${VALUE_E}$ -l ${VALUE_L} -z ${VALUE_Z} -h ${VALUE_H} -j ${VALUE_J} < "${VALUE_P}/${VALUE_S}_data.in"
  fi
elif [ "${VALUE_M}" = "randomforest" ] || [ "${VALUE_M}" = "r" ]; then
  if [ "${FLG_W}" = "TRUE" ]; then
    script/build/pipeline_1_training.out -f ${VALUE_P} -p ${VALUE_S} -r ${VALUE_R} -h ${VALUE_H} -j ${VALUE_J} -c ${VALUE_C} -e ${VALUE_E}$ -l ${VALUE_L} -z ${VALUE_Z} -w < "${VALUE_P}/${VALUE_S}_data.in"
  else
    script/build/pipeline_1_training.out -f ${VALUE_P} -p ${VALUE_S} -r ${VALUE_R} -h ${VALUE_H} -j ${VALUE_J} -c ${VALUE_C} -e ${VALUE_E}$ -l ${VALUE_L} -z ${VALUE_Z} < "${VALUE_P}/${VALUE_S}_data.in"
  fi
else
  echo "m=${VALUE_M} is not supported"
fi

if [ -e "${VALUE_P}/${VALUE_S}_communities.out" ]; then
  echo "Start Clustering trial=${VALUE_S}"
else
  echo "Community detection failed trial=${VALUE_S}. Switch to epsilon=1.0."
  if [ "${VALUE_M}" = "xgboost" ] || [ "${VALUE_M}" = "x" ]; then
    if [ "${FLG_W}" = "TRUE" ]; then
      script/build/pipeline_1_training.out -f ${VALUE_P} -p ${VALUE_S} -r ${VALUE_R} -c ${VALUE_C} -e ${VALUE_E}$ -l 1.0 -z ${VALUE_Z} -h ${VALUE_H} -j ${VALUE_J} -w < "${VALUE_P}/${VALUE_S}_data.in"
    else
      script/build/pipeline_1_training.out -f ${VALUE_P} -p ${VALUE_S} -r ${VALUE_R} -c ${VALUE_C} -e ${VALUE_E}$ -l 1.0-z ${VALUE_Z} -h ${VALUE_H} -j ${VALUE_J} < "${VALUE_P}/${VALUE_S}_data.in"
    fi
  elif [ "${VALUE_M}" = "randomforest" ] || [ "${VALUE_M}" = "r" ]; then
    if [ "${FLG_W}" = "TRUE" ]; then
      script/build/pipeline_1_training.out -f ${VALUE_P} -p ${VALUE_S} -r ${VALUE_R} -h ${VALUE_H} -j ${VALUE_J} -c ${VALUE_C} -e ${VALUE_E}$ -l 1.0 -z ${VALUE_Z} -w < "${VALUE_P}/${VALUE_S}_data.in"
    else
      script/build/pipeline_1_training.out -f ${VALUE_P} -p ${VALUE_S} -r ${VALUE_R} -h ${VALUE_H} -j ${VALUE_J} -c ${VALUE_C} -e ${VALUE_E}$ -l 1.0 -z ${VALUE_Z} < "${VALUE_P}/${VALUE_S}_data.in"
    fi
  else
    echo "m=${VALUE_M} is not supported"
  fi
fi

python3 script/pipeline_3_clustering.py -p "${VALUE_P}/${VALUE_S}_data.in" -q "${VALUE_P}/${VALUE_S}_communities.out" -k ${VALUE_K} -s ${VALUE_S} > "${VALUE_P}/${VALUE_S}_leak.csv"
echo "Clustering is complete trial=${VALUE_S}"