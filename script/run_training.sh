while getopts d:p:i:r:c:j:e:w OPT
do
  case $OPT in
    "d" ) FLG_D="TRUE" ; VALUE_D="$OPTARG" ;;
    "p" ) FLG_P="TRUE" ; VALUE_P="$OPTARG" ;;
    "i" ) FLG_I="TRUE" ; VALUE_I="$OPTARG" ;;
    "r" ) FLG_R="TRUE" ; VALUE_R="$OPTARG" ;;
    "c" ) FLG_C="TRUE" ; VALUE_C="$OPTARG" ;;
    "j" ) FLG_J="TRUE" ; VALUE_J="$OPTARG" ;;
    "e" ) FLG_E="TRUE" ; VALUE_E="$OPTARG" ;;
    "w" ) FLG_W="TRUE" ; VALUE_W="$OPTARG" ;;
  esac
done

g++ -pthread -o script/build/pipeline_1_training.out script/pipeline_1_training.cpp
g++ -o script/build/pipeline_2_louvain.out script/pipeline_2_louvain.cpp

for i in $(seq 1 5)
do 
    echo "random seed is $i"
    python3 ./data/prep.py -d ${VALUE_D} -p "./data/${VALUE_D}/" -i ${VALUE_I} -s ${i}
    cp "./data/${VALUE_D}/${VALUE_D}.in" "${VALUE_P}/${VALUE_D}_${i}.in"
    if [ "${FLG_W}" = "TRUE" ]; then
      script/build/pipeline_1_training.out -f ${VALUE_P} -p ${i} -r ${VALUE_R} -c ${VALUE_C} -j ${VALUE_J} -w < "./data/${VALUE_D}/${VALUE_D}.in"
    else
      script/build/pipeline_1_training.out -f ${VALUE_P} -p ${i} -r ${VALUE_R} -c ${VALUE_C} -j ${VALUE_J} < "./data/${VALUE_D}/${VALUE_D}.in"
    fi
    script/build/pipeline_2_louvain.out -c ${VALUE_C} -e ${VALUE_E}$ < "${VALUE_P}/${i}_adj_mat.txt" > "${VALUE_P}/${i}_communities.out"
done