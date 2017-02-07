//
// Created by lt on 6/2/17.
//

#include <iostream>
#include "LightLDAWrapper.h"

int main()
{
    std::string data_path;
    std::string corpus_path;
    std::string light_lda_path;
    corpus_path = "/home/lt/quanox/First_project/data/tf.dat";
    data_path = "/home/lt/quanox/First_project/data/";
    light_lda_path = "/home/lt/recommendation-python/Profiling_for_rtl/gender_prediction/lightlda/bin/";
    LightLDAWrapper llda;
//    llda.generate_libsvm_inputs(corpus_path, data_path);
//    llda.generate_binary_from_libsvm(data_path, light_lda_path);
//    llda.apply_lda_on_binary(data_path, light_lda_path);
    llda.get_gamma_lambda(light_lda_path);
    return 0;
}

