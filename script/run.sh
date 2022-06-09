while getopts d:r:c: OPT
do
  case $OPT in
    "d" ) FLG_D="TRUE" ; VALUE_D="$OPTARG" ;;
    "r" ) FLG_R="TRUE" ; VALUE_R="$OPTARG" ;;
    "c" ) FLG_C="TRUE" ; VALUE_C="$OPTARG" ;;
  esac
done

TEMPD=$(mktemp -d -t ci-$(date +%Y-%m-%d-%H-%M-%S)-XXXXXXXXXX --tmpdir=result)

script/run_training.sh -d ${VALUE_D} -p ${TEMPD} -r ${VALUE_R} -c ${VALUE_C} > "${TEMPD}/result.ans"
script/run_extract_result.sh -o ${TEMPD}
python3 script/clustering.py -p ${TEMPD}
# python3 script/network_analysis.py -p ${TEMPD}
python3 script/make_report.py -p ${TEMPD} > "${TEMPD}/report.out"