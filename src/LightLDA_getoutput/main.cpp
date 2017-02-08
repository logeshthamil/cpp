//
// Created by lt on 8/2/17.
//

#include <iostream>
#include <vector>
#include <map>
#include <algorithm>
#include "LightLDA_getoutput.h"

class CLParser {
public:

    CLParser(int argc_, char *argv_[], bool switches_on_ = false);

    ~CLParser() {}

    std::string get_arg(std::string s);

private:

    int argc;
    std::vector<std::string> argv;

    bool switches_on;
    std::map<std::string, std::string> switch_map;
};

CLParser::CLParser(int argc_, char *argv_[], bool switches_on_) {
    argc = argc_;
    argv.resize(argc);
    copy(argv_, argv_ + argc, argv.begin());
    switches_on = switches_on_;

    //map the switches to the actual
    //arguments if necessary
    if (switches_on) {
        std::vector<std::string>::iterator it1, it2;
        it1 = argv.begin();
        it2 = it1 + 1;

        while (true) {
            if (it1 == argv.end()) break;
            if (it2 == argv.end()) break;

            if ((*it1)[0] == '-')
                switch_map[*it1] = *(it2);

            it1++;
            it2++;
        }
    }
}


std::string CLParser::get_arg(std::string s) {
    if (!switches_on) return "";

    if (switch_map.find(s) != switch_map.end())
        return switch_map[s];

    return "";
}


int main(int argc, char *argv[]) {
    std::string data_path;
    std::string corpus_path;
    std::string light_lda_path;

    CLParser cmd_line(argc, argv, true);

    std::string temp;

    auto a = cmd_line.get_arg("-corpus_file");
    if (a != "") corpus_path = temp;

    auto b = cmd_line.get_arg("-lda_path");
    if (b != "") light_lda_path = temp;

    auto c = cmd_line.get_arg("-output_path");
    if (c != "") data_path = temp;

//    a = "/home/lt/quanox/First_project/data/tf_test.dat";
//    c = "/home/lt/quanox/First_project/data/";
//    b = "/home/lt/recommendation-python/Profiling_for_rtl/gender_prediction/lightlda/bin/";

    if (a != "" && b != "" && c != "") {
        LightLDA_getoutput llda;
        llda.generate_libsvm_inputs(a, c);
        llda.generate_binary_from_libsvm(c, b);
        llda.apply_lda_on_binary(c, b);
        llda.get_gamma_lambda(c);
    } else {
        std::cout << "Please give the valid parameters" << std::endl;
    }
    return 0;

}

// "Usage"   ./get_output -corpus_file "/home/lt/quanox/First_project/data/tf_output.dat" -lda_path "/home/lt/recommendation-python/Profiling_for_rtl/gender_prediction/lightlda/bin/" -output_path "/home/lt/quanox/First_project/data/"
