while getopts d:r:c:i:e:wn OPT
do
  case $OPT in
    "d" ) FLG_D="TRUE" ; VALUE_D="$OPTARG" ;;
    "r" ) FLG_R="TRUE" ; VALUE_R="$OPTARG" ;;
    "c" ) FLG_C="TRUE" ; VALUE_C="$OPTARG" ;;
    "i" ) FLG_I="TRUE" ; VALUE_I="$OPTARG" ;;
    "e" ) FLG_E="TRUE" ; VALUE_E="$OPTARG" ;;
    "w" ) FLG_W="TRUE" ; VALUE_W="$OPTARG" ;;
    "n" ) FLG_N="TRUE" ; VALUE_N="$OPTARG" ;;
  esac
done

TEMPD=$(mktemp -d -t ci-$(date +%Y-%m-%d-%H-%M-%S)-XXXXXXXXXX --tmpdir=result)

echo -e "d,${VALUE_D}\nr,${VALUE_R}\nc,${VALUE_C}\ni,${VALUE_I}\ne,${VALUE_E}\nw,${FLG_W}\nn,${FLG_N}" > "${TEMPD}/param.csv"

if [ "${FLG_W}" = "TRUE" ]; then
  script/run_training.sh -d ${VALUE_D} -p ${TEMPD} -r ${VALUE_R} -c ${VALUE_C} -i ${VALUE_I} -e ${VALUE_E}$ -w
else
  script/run_training.sh -d ${VALUE_D} -p ${TEMPD} -r ${VALUE_R} -c ${VALUE_C} -i ${VALUE_I} -e ${VALUE_E}$
fi

script/run_extract_result.sh -o ${TEMPD}
python3 script/pipeline_3_clustering.py -p ${TEMPD} > "${TEMPD}/leak_f1.csv"

if [ "${FLG_N}" = "TRUE" ]; then
  python3 script/pipeline_4_vis_network.py -p ${TEMPD}
fi

python3 script/pipeline_5_report.py -p ${TEMPD} > "${TEMPD}/report.md"

rm ${TEMPD}/*.in
rm ${TEMPD}/*.txt
rm ${TEMPD}/*.out