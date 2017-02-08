//
// Created by lt on 6/2/17.
//

#ifndef FIRST_PROJECT_LIGHTLDAWRAPPER_H
#define FIRST_PROJECT_LIGHTLDAWRAPPER_H
#include <string>


class LightLDAWrapper {
public:
    void generate_libsvm_inputs(std::string corpus_path, std::string output_data_path);
    void generate_binary_from_libsvm(std::string output_data_path, std::string lda_path);
    void apply_lda_on_binary(std::string output_data_path, std::string lda_path);

    void get_gamma_lambda(std::string output_data_path); // conditional probablity table

};


#endif //FIRST_PROJECT_LIGHTLDAWRAPPER_H
