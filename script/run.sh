VALUE_D="breastcancer"
VALUE_R=20
VALUE_C=1
VALUE_J=1
VALUE_N=20000
VALUE_I=1
VALUE_E=0.3
VALUE_T="result"

while getopts d:r:c:j:n:i:e:t:wg OPT
do
  case $OPT in
    "d" ) FLG_D="TRUE" ; VALUE_D="$OPTARG" ;;
    "r" ) FLG_R="TRUE" ; VALUE_R="$OPTARG" ;;
    "c" ) FLG_C="TRUE" ; VALUE_C="$OPTARG" ;;
    "j" ) FLG_J="TRUE" ; VALUE_J="$OPTARG" ;;
    "n" ) FLG_N="TRUE" ; VALUE_N="$OPTARG" ;;
    "i" ) FLG_I="TRUE" ; VALUE_I="$OPTARG" ;;
    "e" ) FLG_E="TRUE" ; VALUE_E="$OPTARG" ;;
    "t" ) FLG_T="TRUE" ; VALUE_T="$OPTARG" ;;
    "w" ) FLG_W="TRUE" ; VALUE_W="$OPTARG" ;;
    "g" ) FLG_G="TRUE" ; VALUE_G="$OPTARG" ;;
  esac
done

TEMPD=$(mktemp -d -t ci-$(date +%Y-%m-%d-%H-%M-%S)-XXXXXXXXXX --tmpdir=${VALUE_T})

echo -e "d,${VALUE_D}\nr,${VALUE_R}\nc,${VALUE_C}\ni,${VALUE_I}\ne,${VALUE_E}\nw,${FLG_W}\nn,${VALUE_N}" > "${TEMPD}/param.csv"

if [ "${FLG_W}" = "TRUE" ]; then
  script/run_training.sh -d ${VALUE_D} -p ${TEMPD} -r ${VALUE_R} -c ${VALUE_C} -j ${VALUE_J} -n ${VALUE_N} -i ${VALUE_I} -e ${VALUE_E}$ -w
else
  script/run_training.sh -d ${VALUE_D} -p ${TEMPD} -r ${VALUE_R} -c ${VALUE_C} -j ${VALUE_J} -n ${VALUE_N} -i ${VALUE_I} -e ${VALUE_E}$
fi

script/run_extract_result.sh -o ${TEMPD}
python3 script/pipeline_3_clustering.py -p ${TEMPD} > "${TEMPD}/leak_vmeasure.csv"

if [ "${FLG_G}" = "TRUE" ]; then
  python3 script/pipeline_4_vis_network.py -p ${TEMPD} -e ${VALUE_E}
fi

python3 script/pipeline_5_report.py -p ${TEMPD} > "${TEMPD}/report.md"

rm ${TEMPD}/*.in
rm ${TEMPD}/*.txt
rm ${TEMPD}/*.out
