while getopts d:m:p:n:f:v:i:r:c:a:h:b:j:e:l:o:z:k:s:w:x:g OPT; do
    case $OPT in
    "d")
        FLG_D="TRUE"
        VALUE_D="$OPTARG"
        ;;
    "m")
        FLG_M="TRUE"
        VALUE_M="$OPTARG"
        ;;
    "p")
        FLG_P="TRUE"
        VALUE_P="$OPTARG"
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
    "s")
        FLG_S="TRUE"
        VALUE_S="$OPTARG"
        ;;
    "w")
        FLG_W="TRUE"
        VALUE_W="$OPTARG"
        ;;
    "x")
        FLG_X="TRUE"
        VALUE_X="$OPTARG"
        ;;
    "g")
        FLG_G="TRUE"
        VALUE_G="$OPTARG"
        ;;
    esac
done

echo "random seed is ${VALUE_S}"
python3 ./data/prep.py -d ${VALUE_D} -p "./data/${VALUE_D}/" -n ${VALUE_N} -f ${VALUE_F} -v ${VALUE_V} -i ${VALUE_I} -s ${VALUE_S}
cp "./data/${VALUE_D}/${VALUE_D}_${VALUE_S}.in" "${VALUE_P}/${VALUE_S}_data.in"

for TEMP_VALUE_L in ${VALUE_L} 0.1 1.0; do
    echo "epsilon=${TEMP_VALUE_L} trial=${VALUE_S}"

    RUNCMD="build/script/pipeline_1_training.out -f ${VALUE_P} -p ${VALUE_S} -r ${VALUE_R} -h ${VALUE_H} -b ${VALUE_B} -j ${VALUE_J} -c ${VALUE_C} -e ${VALUE_E} -l ${TEMP_VALUE_L} -o ${VALUE_O} -z ${VALUE_Z} -w ${VALUE_W} -x ${VALUE_X}"
    if [ "${VALUE_M}" = "xgboost" ] || [ "${VALUE_M}" = "x" ] || [ "${VALUE_M}" = "secureboost" ] || [ "${VALUE_M}" = "s" ]; then
        RUNCMD+=" -a ${VALUE_A}"
    fi
    if [ "${FLG_G}" = "TRUE" ]; then
        RUNCMD+=" -g"
    fi

    eval ${RUNCMD} <"${VALUE_P}/${VALUE_S}_data.in"

    if [ -e "${VALUE_P}/${VALUE_S}_clusters_and_labels.out" ]; then
        break
    fi
done
