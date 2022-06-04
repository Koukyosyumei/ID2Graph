while getopts d: OPT
do
  case $OPT in
    "d" ) FLG_D="TRUE" ; VALUE_D="$OPTARG" ;;
  esac
done

TEMPD=$(mktemp -d -t ci-$(date +%Y-%m-%d-%H-%M-%S)-XXXXXXXXXX --tmpdir=result)

script/run_training.sh -d ${VALUE_D} -p ${TEMPD} > "${TEMPD}/result.ans"
script/run_extract_result.sh -o ${TEMPD}
python3 script/make_report.py -p ${TEMPD} > "${TEMPD}/report.out"