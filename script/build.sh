VALUE_J=1

while getopts j: OPT; do
  case $OPT in
  "j")
    FLG_J="TRUE"
    VALUE_J="$OPTARG"
    ;;
  esac
done

cmake -S . -B build
cmake --build build -j ${VALUE_J}
