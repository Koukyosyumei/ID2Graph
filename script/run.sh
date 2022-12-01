#!/bin/bash

# constant values
NUM_TRIAL=5

# default values
VALUE_D="breastcancer"
VALUE_M="xgboost"
VALUE_R=20
VALUE_C=1
VALUE_A=0.3
VALUE_H=3
VALUE_B=-1
VALUE_J=1
VALUE_N=20000
VALUE_F=0.5
VALUE_V=-1
VALUE_I=1
VALUE_E=0.3
VALUE_K=1.0
VALUE_T="result/temp"
VALUE_U="result"
VALUE_P=1
VALUE_L=10
VALUE_Z=300
VALUE_O=-1
VALUE_W=-1
VALUE_X=2

while getopts d:m:r:c:a:h:j:n:f:v:i:e:l:o:z:t:u:p:b:w:x:k:ygq OPT; do
  case $OPT in
  "d")
    FLG_D="TRUE"
    VALUE_D="$OPTARG"
    ;;
  "m")
    FLG_M="TRUE"
    VALUE_M="$OPTARG"
    ;;
  "r")
    FLG_R="TRUE"
    VALUE_R="$OPTARG"
    ;;
  "c")
    FLG_C="TRUE"
    VALUE_C="$OPTARG"
    ;;
  "a")
    FLG_A="TRUE"
    VALUE_A="$OPTARG"
    ;;
  "h")
    FLG_H="TRUE"
    VALUE_H="$OPTARG"
    ;;
  "b")
    FLG_B="TRUE"
    VALUE_B="$OPTARG"
    ;;
  "j")
    FLG_J="TRUE"
    VALUE_J="$OPTARG"
    ;;
  "n")
    FLG_N="TRUE"
    VALUE_N="$OPTARG"
    ;;
  "f")
    FLG_F="TRUE"
    VALUE_F="$OPTARG"
    ;;
  "v")
    FLG_V="TRUE"
    VALUE_V="$OPTARG"
    ;;
  "i")
    FLG_I="TRUE"
    VALUE_I="$OPTARG"
    ;;
  "e")
    FLG_E="TRUE"
    VALUE_E="$OPTARG"
    ;;
  "l")
    FLG_L="TRUE"
    VALUE_L="$OPTARG"
    ;;
  "o")
    FLG_O="TRUE"
    VALUE_O="$OPTARG"
    ;;
  "z")
    FLG_Z="TRUE"
    VALUE_Z="$OPTARG"
    ;;
  "k")
    FLG_K="TRUE"
    VALUE_K="$OPTARG"
    ;;
  "t")
    FLG_T="TRUE"
    VALUE_T="$OPTARG"
    ;;
  "u")
    FLG_U="TRUE"
    VALUE_U="$OPTARG"
    ;;
  "p")
    FLG_P="TRUE"
    VALUE_P="$OPTARG"
    ;;
  "w")
    FLG_W="TRUE"
    VALUE_W="$OPTARG"
    ;;
  "x")
    FLG_X="TRUE"
    VALUE_X="$OPTARG"
    ;;
  "y")
    FLG_Y="TRUE"
    VALUE_Y="$OPTARG"
    ;;
  "g")
    FLG_G="TRUE"
    VALUE_G="$OPTARG"
    ;;
  "q")
    FLG_Q="TRUE"
    VALUE_Q="$OPTARG"
    ;;
  esac
done

RESUD=$(mktemp -d -t ci-$(date +%Y-%m-%d-%H-%M-%S)-XXXXXXXXXX --tmpdir=${VALUE_U})
TEMPD=$(mktemp -d -t ci-$(date +%Y-%m-%d-%H-%M-%S)-XXXXXXXXXX --tmpdir=${VALUE_T})

echo -e "d,${VALUE_D}\nm,${VALUE_M}\nr,${VALUE_R}\nc,${VALUE_C}\na,${VALUE_A}\nh,${VALUE_H}\nb,${VALUE_B}\ni,${VALUE_I}\ne,${VALUE_E}\nl,${VALUE_L}\no,${VALUE_O}\nw,${FLG_W}\nn,${VALUE_N}\nf,${VALUE_F}\nv,${VALUE_V}\nk,${VALUE_K}\nx,${VALUE_X}\ny,${FLG_Y}" >"${RESUD}/param.csv"

if [ "${VALUE_M}" = "xgboost" ] || [ "${VALUE_M}" = "x" ]; then
  cp build/script/train_xgboost build/script/pipeline_1_training.out
elif [ "${VALUE_M}" = "secureboost" ] || [ "${VALUE_M}" = "s" ]; then
  cp build/script/train_secureboost build/script/pipeline_1_training.out
elif [ "${VALUE_M}" = "randomforest" ] || [ "${VALUE_M}" = "r" ]; then
  cp build/script/train_randomforest build/script/pipeline_1_training.out
else
  echo "m=${VALUE_M} is not supported"
fi

for s in $(seq 1 ${NUM_TRIAL}); do
  TRAINCMD="script/run_training.sh -s ${s} -d ${VALUE_D} -m ${VALUE_M} -p ${TEMPD} -r ${VALUE_R} -c ${VALUE_C} -a ${VALUE_A} -h ${VALUE_H} -b ${VALUE_B} -j ${VALUE_J} -n ${VALUE_N} -f ${VALUE_F} -v ${VALUE_V} -i ${VALUE_I} -e ${VALUE_E} -l ${VALUE_L} -o ${VALUE_O} -z ${VALUE_Z} -k ${VALUE_K} -w ${VALUE_W} -x ${VALUE_X}"
  if [ "${FLG_Y}" = "TRUE" ]; then
    TRAINCMD+=" -y"
  fi
  if [ "${FLG_G}" = "TRUE" ]; then
    TRAINCMD+=" -g"
  fi
  if [ "${FLG_Q}" = "TRUE" ]; then
    TRAINCMD+=" -q"
  fi
  if [ ${VALUE_P} -gt 1 ]; then
    if [ $((${s} % ${VALUE_P})) -ne 0 ] && [ ${s} -ne ${NUM_TRIAL} ]; then
      TRAINCMD+=" &"
    else
      TRAINCMD+=" & wait"
    fi
  fi
  eval ${TRAINCMD}
done

script/run_extract_result.sh -o ${TEMPD}

if [ "${FLG_G}" = "TRUE" ]; then
  echo "Drawing a network ..."
  python3 script/pipeline_3_vis_network.py -p ${TEMPD} -e ${VALUE_E}
fi

echo "Making a report ..."
python3 script/pipeline_4_report.py -p ${TEMPD} >"${RESUD}/report.md"

mv ${TEMPD}/*.ans ${RESUD}/
mv ${TEMPD}/leak.csv ${RESUD}/
mv ${TEMPD}/loss_lp.csv ${RESUD}/
mv ${TEMPD}/result.png ${RESUD}/

if [ "${FLG_Q}" = "TRUE" ]; then
  mv ${TEMPD}/*.html ${RESUD}/
fi

for s in $(seq 1 ${NUM_TRIAL}); do
  if [ -e ${TEMPD}/${s}_adj_mat_plot.png ]; then
    mv ${TEMPD}/${s}_adj_mat_plot.png ${RESUD}/
  fi
done

wait
rm -rf ${TEMPD}
