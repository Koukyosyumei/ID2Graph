[1mdiff --git a/secureboost/node.h b/secureboost/node.h[m
[1mindex 0e6b799..83b0fb6 100644[m
[1m--- a/secureboost/node.h[m
[1m+++ b/secureboost/node.h[m
[36m@@ -413,10 +413,10 @@[m [mstruct Node[m
 [m
     string recursive_print(string prefix, bool isleft, bool show_purity, bool binary_color)[m
     {[m
[31m-        string node_info = "";[m
[32m+[m[32m        string node_info;[m
         if (is_leaf())[m
         {[m
[31m-            node_info += to_string(get_val());[m
[32m+[m[32m            node_info = to_string(get_val());[m
             vector<int> temp_idxs = get_idxs();[m
             if (show_purity) {[m
                 int cnt_idxs = temp_idxs.size();[m
[36m@@ -433,7 +433,20 @@[m [mstruct Node[m
                     double purity = max(double(cnt_zero) / double(cnt_idxs),[m
                                         1 - double(cnt_zero)/double(cnt_idxs));[m
                     node_info += ", ";[m
[31m-                    node_info += to_string(purity);[m
[32m+[m
[32m+[m[32m                    if (binary_color){[m
[32m+[m[32m                        if (purity < 0.6){[m
[32m+[m[32m                            node_info += "\033[32m";[m
[32m+[m[32m                        } else if (purity < 0.8){[m
[32m+[m[32m                            node_info += "\033[33m";[m
[32m+[m[32m                        } else {[m
[32m+[m[32m                            node_info += "\033[31m";[m
[32m+[m[32m                        }[m
[32m+[m[32m                        node_info += to_string(purity);[m
[32m+[m[32m                        node_info += "\033[0m";[m
[32m+[m[32m                    } else {[m
[32m+[m[32m                        node_info += to_string(purity);[m
[32m+[m[32m                    }[m
                 }[m
             }[m
             else {[m
[36m@@ -471,16 +484,8 @@[m [mstruct Node[m
             node_info += to_string(get_record_id());[m
         }[m
 [m
[31m-        if (isleft)[m
[31m-        {[m
[31m-            node_info = prefix + "├──" + node_info;[m
[31m-            node_info += "\n";[m
[31m-        }[m
[31m-        else[m
[31m-        {[m
[31m-            node_info = prefix + "└──" + node_info;[m
[31m-            node_info += "\n";[m
[31m-        }[m
[32m+[m[32m        node_info = prefix + "|--" + node_info;[m
[32m+[m[32m        node_info += "\n";[m
 [m
         if (!is_leaf())[m
         {[m
