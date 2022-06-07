while getopts d:p: OPT
do
  case $OPT in
    "d" ) FLG_D="TRUE" ; VALUE_D="$OPTARG" ;;
    "p" ) FLG_P="TRUE" ; VALUE_P="$OPTARG" ;;
  esac
done

g++ -o pipeline.out pipeline.cpp
g++ -o pipeline_louvain.out pipeline_louvain.cpp

for i in $(seq 1 5)
do 
    echo "random seed is $i"
    python3 ./data/prep.py -d ${VALUE_D} -p "./data/${VALUE_D}/" -s $i
    cp "data/${VALUE_D}/${VALUE_D}.in" "${VALUE_P}/${VALUE_D}_${i}.in"
    ./pipeline.out ${VALUE_P} ${i} < "data/${VALUE_D}/${VALUE_D}.in"
    ./pipeline_louvain.out < "${VALUE_P}/${i}_adj_mat.txt" > "${VALUE_P}/${i}_communities.out"
done