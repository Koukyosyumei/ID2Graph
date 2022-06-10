while getopts d:r:c:i:wn OPT
do
  case $OPT in
    "d" ) FLG_D="TRUE" ; VALUE_D="$OPTARG" ;;
    "r" ) FLG_R="TRUE" ; VALUE_R="$OPTARG" ;;
    "c" ) FLG_C="TRUE" ; VALUE_C="$OPTARG" ;;
    "i" ) FLG_I="TRUE" ; VALUE_I="$OPTARG" ;;
    "w" ) FLG_W="TRUE" ; VALUE_W="$OPTARG" ;;
    "n" ) FLG_N="TRUE" ; VALUE_N="$OPTARG" ;;
  esac
done

TEMPD=$(mktemp -d -t ci-$(date +%Y-%m-%d-%H-%M-%S)-XXXXXXXXXX --tmpdir=result)

if [ "${FLG_W}" = "TRUE" ]; then
  script/run_training.sh -d ${VALUE_D} -p ${TEMPD} -r ${VALUE_R} -c ${VALUE_C} -i ${VALUE_I} -w > "${TEMPD}/result.ans"
else
  script/run_training.sh -d ${VALUE_D} -p ${TEMPD} -r ${VALUE_R} -c ${VALUE_C} -i ${VALUE_I} > "${TEMPD}/result.ans"
fi

script/run_extract_result.sh -o ${TEMPD}
python3 script/pipeline_3_clustering.py -p ${TEMPD} > "${TEMPD}/leak_f1.csv"

if [ "${FLG_N}" = "TRUE" ]; then
  python3 script/pipeline_4_vis_network.py -p ${TEMPD}
fi

python3 script/pipeline_5_report.py -p ${TEMPD} > "${TEMPD}/report.out"