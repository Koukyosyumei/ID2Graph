#!/bin/bash

# constant values
NUM_TRIAL=5

# default values
VALUE_D="breastcancer"
VALUE_M="xgboost"
VALUE_R=20
VALUE_C=1
VALUE_H=3
VALUE_J=1
VALUE_N=20000
VALUE_F=0.5
VALUE_I=1
VALUE_E=0.3
VALUE_K="vanila"
VALUE_T="result/temp"
VALUE_U="result"
VALUE_P=1

while getopts d:m:r:c:h:j:n:f:i:e:t:u:p:wg OPT
do
  case $OPT in
    "d" ) FLG_D="TRUE" ; VALUE_D="$OPTARG" ;;
    "m" ) FLG_M="TRUE" ; VALUE_M="$OPTARG" ;;
    "r" ) FLG_R="TRUE" ; VALUE_R="$OPTARG" ;;
    "c" ) FLG_C="TRUE" ; VALUE_C="$OPTARG" ;;
    "h" ) FLG_H="TRUE" ; VALUE_H="$OPTARG" ;;
    "j" ) FLG_J="TRUE" ; VALUE_J="$OPTARG" ;;
    "n" ) FLG_N="TRUE" ; VALUE_N="$OPTARG" ;;
    "f" ) FLG_F="TRUE" ; VALUE_F="$OPTARG" ;;
    "i" ) FLG_I="TRUE" ; VALUE_I="$OPTARG" ;;
    "e" ) FLG_E="TRUE" ; VALUE_E="$OPTARG" ;;
    "k" ) FLG_K="TRUE" ; VALUE_K="$OPTARG" ;;
    "t" ) FLG_T="TRUE" ; VALUE_T="$OPTARG" ;;
    "u" ) FLG_U="TRUE" ; VALUE_U="$OPTARG" ;;
    "p" ) FLG_P="TRUE" ; VALUE_P="$OPTARG" ;;
    "w" ) FLG_W="TRUE" ; VALUE_W="$OPTARG" ;;
    "g" ) FLG_G="TRUE" ; VALUE_G="$OPTARG" ;;
  esac
done

RESUD=$(mktemp -d -t ci-$(date +%Y-%m-%d-%H-%M-%S)-XXXXXXXXXX --tmpdir=${VALUE_U})
TEMPD=$(mktemp -d -t ci-$(date +%Y-%m-%d-%H-%M-%S)-XXXXXXXXXX --tmpdir=${VALUE_T})

echo -e "d,${VALUE_D}\nm,${VALUE_M}\nr,${VALUE_R}\nc,${VALUE_C}\nh,${VALUE_H}\ni,${VALUE_I}\ne,${VALUE_E}\nw,${FLG_W}\nn,${VALUE_N}\nf,${VALUE_F}\nk,${VALUE_K}" > "${RESUD}/param.csv"

if [ "${VALUE_M}" = "xgboost" ] || [ "${VALUE_M}" = "x" ]; then
  g++ -pthread -o script/build/pipeline_1_training.out script/pipeline_1_train_xgboost.cpp
  g++ -o script/build/pipeline_2_louvain.out script/pipeline_2_louvain.cpp
elif [ "${VALUE_M}" = "randomforest" ] || [ "${VALUE_M}" = "r" ]; then
  g++ -pthread -o script/build/pipeline_1_training.out script/pipeline_1_train_randomforest.cpp
  g++ -o script/build/pipeline_2_louvain.out script/pipeline_2_louvain.cpp
else
  echo "m=${VALUE_M} is not supported"
fi


for s in $(seq 1 ${NUM_TRIAL})
do 
  TRAINCMD="script/run_training.sh -s ${s} -d ${VALUE_D} -m ${VALUE_M} -p ${TEMPD} -r ${VALUE_R} -c ${VALUE_C} -h ${VALUE_H} -j ${VALUE_J} -n ${VALUE_N} -f ${VALUE_F} -i ${VALUE_I} -e ${VALUE_E} -k ${VALUE_K}"
  if [ "${FLG_W}" = "TRUE" ]; then
    TRAINCMD+=" -w"
  fi
  if [ $((${s} % ${VALUE_P})) -ne 0 ] && [ ${s} -ne ${NUM_TRIAL} ]; then
    TRAINCMD+=" &"
  fi
  echo ${TRAINCMD}
  eval ${TRAINCMD}
done

script/run_extract_result.sh -o ${TEMPD}

if [ "${FLG_G}" = "TRUE" ]; then
  python3 script/pipeline_4_vis_network.py -p ${TEMPD} -e ${VALUE_E}
fi

python3 script/pipeline_5_report.py -p ${TEMPD} > "${RESUD}/report.md"

mv ${TEMPD}/*.ans ${RESUD}/
mv ${TEMPD}/leak.csv ${RESUD}/
mv ${TEMPD}/loss_lp.csv ${RESUD}/
mv ${TEMPD}/result.png ${RESUD}/

rm -rf ${TEMPD}
