#include "llatvfl/attack/attack.h"
#include "llatvfl/attack/baseline.h"
#include "llatvfl/louvain/louvain.h"
#include "llatvfl/lpmst/lpmst.h"
#include "llatvfl/paillier/keygenerator.h"
#include "llatvfl/utils/metric.h"
#include <cassert>
#include <chrono>
#include <fstream>
#include <future>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>
using namespace std;

const int n_job = 1;
const int max_bin = 32;
const float lam = 1.0;
const float const_gamma = 0.0;
const float eps = 1.0;
const float min_child_weight = -1 * numeric_limits<float>::infinity();
const float subsample_cols = 0.8;
const bool use_missing_value = false;
const int m_lpmst = 2;

string folderpath;
string fileprefix;
int boosting_rounds = 20;
int completely_secure_round = 0;
int depth = 3;
int min_leaf = 1;
float learning_rate = 0.3;
float mi_bound = numeric_limits<float>::infinity();
float eta = 0.3;
float epsilon_ldp = -1;
int maximum_nb_pass_done = 300;
bool save_adj_mat = false;
bool is_freerider = false;
bool use_uniontree = false;
int max_num_samples_in_a_chunk = 1000000;
int edge_weight_between_chunks = 100;

void parse_args(int argc, char *argv[]) {
  int opt;
  while ((opt = getopt(argc, argv, "f:p:r:c:a:e:h:j:l:o:b:w:y:xgq")) != -1) {
    switch (opt) {
    case 'f':
      folderpath = string(optarg);
      break;
    case 'p':
      fileprefix = string(optarg);
      break;
    case 'r':
      boosting_rounds = stoi(string(optarg));
      break;
    case 'c':
      completely_secure_round = stoi(string(optarg));
      break;
    case 'a':
      learning_rate = stof(string(optarg));
      break;
    case 'e':
      eta = stof(string(optarg));
      break;
    case 'h':
      depth = stoi(string(optarg));
      break;
    case 'j':
      min_leaf = stoi(string(optarg));
      break;
    case 'l':
      maximum_nb_pass_done = stoi(string(optarg));
      break;
    case 'o':
      epsilon_ldp = stof(string(optarg));
      break;
    case 'b':
      mi_bound = stof(string(optarg));
      break;
    case 'w':
      max_num_samples_in_a_chunk = stoi(string(optarg));
      break;
    case 'y':
      edge_weight_between_chunks = stoi(string(optarg));
      break;
    case 'x':
      use_uniontree = true;
      break;
    case 'g':
      save_adj_mat = true;
      break;
    default:
      printf("unknown parameter %s is specified", optarg);
      break;
    }
  }
}

int main(int argc, char *argv[]) {
  parse_args(argc, argv);

  // --- Load Data --- //
  int num_classes, num_row_train, num_row_val, num_col, num_party;
  int num_nan_cell = 0;
  if (scanf("%d %d %d %d", &num_classes, &num_row_train, &num_col,
            &num_party) != 4) {
    try {
      throw runtime_error("bad input");
    } catch (std::runtime_error e) {
      cerr << e.what() << "\n";
    }
  }
  vector<vector<float>> X_train(num_row_train, vector<float>(num_col));
  vector<float> y_train(num_row_train);
  vector<float> y_hat;
  vector<SecureBoostParty> parties(num_party);

  int temp_count_feature = 0;
  for (int i = 0; i < num_party; i++) {
    int num_col = 0;
    if (scanf("%d", &num_col) != 1) {
      try {
        throw runtime_error("bad input");
      } catch (std::runtime_error e) {
        cerr << e.what() << "\n";
      }
    }
    vector<int> feature_idxs(num_col);
    vector<vector<float>> x(num_row_train, vector<float>(num_col));
    for (int j = 0; j < num_col; j++) {
      feature_idxs[j] = temp_count_feature;
      for (int k = 0; k < num_row_train; k++) {
        if (scanf("%f", &x[k][j]) != 1) {
          try {
            throw runtime_error("bad input");
          } catch (std::runtime_error e) {
            cerr << e.what() << "\n";
          }
        }
        X_train[k][temp_count_feature] = x[k][j];
      }
      temp_count_feature += 1;
    }
    SecureBoostParty party(x, num_classes, feature_idxs, i, min_leaf,
                           subsample_cols, max_bin, use_missing_value);
    parties[i] = party;
  }
  for (int j = 0; j < num_row_train; j++) {
    if (scanf("%f", &y_train[j]) != 1) {
      try {
        throw runtime_error("bad input");
      } catch (std::runtime_error e) {
        cerr << e.what() << "\n";
      }
    }
  }

  if (scanf("%d", &num_row_val) != 1) {
    try {
      throw runtime_error("bad input");
    } catch (std::runtime_error e) {
      cerr << e.what() << "\n";
    }
  }
  vector<vector<float>> X_val(num_row_val, vector<float>(num_col));
  vector<float> y_val(num_row_val);
  for (int i = 0; i < num_col; i++) {
    for (int j = 0; j < num_row_val; j++) {
      if (scanf("%f", &X_val[j][i]) != 1) {
        try {
          throw runtime_error("bad input");
        } catch (std::runtime_error e) {
          cerr << e.what() << "\n";
        }
      }
    }
  }
  for (int j = 0; j < num_row_val; j++) {
    if (scanf("%f", &y_val[j]) != 1) {
      try {
        throw runtime_error("bad input");
      } catch (std::runtime_error e) {
        cerr << e.what() << "\n";
      }
    }
  }

  PaillierKeyGenerator keygenerator = PaillierKeyGenerator(128);
  pair<PaillierPublicKey, PaillierSecretKey> keypair =
      keygenerator.generate_keypair();
  PaillierPublicKey pk = keypair.first;
  PaillierSecretKey sk = keypair.second;

  for (int i = 0; i < num_party; i++) {
    parties[i].set_publickey(pk);
  }
  parties[0].set_secretkey(sk);

  std::ofstream result_file;
  string result_filepath = folderpath + "/" + fileprefix + "_result.ans";
  result_file.open(result_filepath, std::ios::out);
  result_file << "train size," << num_row_train << "\n";
  result_file << "val size," << num_row_val << "\n";
  result_file << "column size," << num_col << "\n";
  result_file << "party size," << num_party << "\n";
  result_file << "num of nan," << num_nan_cell << "\n";

  // --- Check Initialization --- //
  SecureBoostClassifier clf = SecureBoostClassifier(
      num_classes, subsample_cols, min_child_weight, depth, min_leaf,
      learning_rate, boosting_rounds, lam, const_gamma, eps, mi_bound, 0,
      completely_secure_round, 1 / num_classes, n_job, true);

  printf("Start training trial=%s\n", fileprefix.c_str());
  chrono::system_clock::time_point start, end;
  start = chrono::system_clock::now();
  if (epsilon_ldp > 0) {
    y_hat.reserve(num_row_train);
    LPMST lp_1st(m_lpmst, epsilon_ldp, 0);
    lp_1st.fit(clf, parties, y_train, y_hat);
  } else {
    clf.fit(parties, y_train);
  }
  end = chrono::system_clock::now();
  float elapsed =
      chrono::duration_cast<chrono::milliseconds>(end - start).count();
  result_file << "training time," << elapsed << "\n";
  printf("Training is complete %f [ms] trial=%s\n", elapsed,
         fileprefix.c_str());

  for (int i = 0; i < clf.logging_loss.size(); i++) {
    result_file << "round " << i + 1 << ": " << clf.logging_loss[i] << "\n";
  }

  for (int i = 0; i < clf.estimators.size(); i++) {
    result_file << "Tree-" << i + 1 << ": "
                << clf.estimators[i].get_leaf_purity() << "\n";
    result_file << clf.estimators[i].print(true, true).c_str() << "\n";
  }

  vector<vector<float>> predict_proba_train = clf.predict_proba(X_train);
  vector<int> y_true_train(y_train.begin(), y_train.end());
  result_file << "Train AUC,"
              << ovr_roc_auc_score(predict_proba_train, y_true_train) << "\n";

  vector<vector<float>> predict_proba_val = clf.predict_proba(X_val);
  vector<int> y_true_val(y_val.begin(), y_val.end());
  result_file << "Val AUC," << ovr_roc_auc_score(predict_proba_val, y_true_val)
              << "\n";

  result_file.close();
}
