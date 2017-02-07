//
// Created by lt on 6/2/17.
//

#include <iostream>
#include <fstream>
#include <map>
#include "LightLDAWrapper.h"
#include <boost/algorithm/string.hpp>
#include "boost/filesystem.hpp"

void get_unique_words(std::string corpus_file, std::string vocab_file)
{
    std::cout << "Writing all the unique words in the file" << std::endl;
    // Open a file stream
    std::fstream fs(corpus_file);
    std::ofstream of(vocab_file);

    // Create a map to store count of all words
    std::map<std::string, int> mp;

    // Keep reading words while there are words to read
    std::string line;
    while (std::getline(fs, line))
    {
        std::vector < std::string > words;
        boost::split(words, line, boost::is_any_of(" "));
        for (auto it :words) {
            ++mp[it];
        }
    }

    fs.close();
    for (std::map <std::string, int> :: iterator p = mp.begin();
         p != mp.end(); p++)
    {
//        std::cout << p->first << "\t" << std::to_string(p->second) << std::endl;
        of << p->first << std::endl;
    }
}

void LightLDAWrapper::generate_libsvm_inputs(std::string corpus_path, std::string output_data_path) {
    std::cout << "Converting the standard corpus to libsvm format to use it as input of light lda" << std::endl;
    std::string libsvm_vocab_path = output_data_path + "libsvm_vocab.dat";
    std::string libsvm_corpus_path = output_data_path + "libsvm_corpus.dat";
    std::fstream corpus_ip(corpus_path);
    std::ofstream libsvmvocab_op(libsvm_vocab_path);
    std::ofstream libsvmcorpus_op(libsvm_corpus_path);
    std::map<std::string, int> vocab_map;
    std::string corpus_line;

    std::cout << "Writing all the vocab into file in libsvm format" << std::endl;
    while (std::getline(corpus_ip, corpus_line))
    {
        std::vector < std::string > words;
        boost::split(words, corpus_line, boost::is_any_of(" "));
        for (auto word :words) {
            ++vocab_map[word];
        }
    }
    corpus_ip.close();

    int temp_i = 0;
    for (std::map <std::string, int> :: iterator p = vocab_map.begin();
         p != vocab_map.end(); p++)
    {
        libsvmvocab_op << temp_i << "\t" << p->first  << "\t" << p->second << std::endl;
        temp_i++;
    }
    libsvmvocab_op.close();

    std::fstream new_corpus_ip(corpus_path);
    temp_i = 0;
    std::vector < std::string > unique_words_inline;

    std::cout << "Writing all the corpus into file in libsvm format" << std::endl;
    while (std::getline(new_corpus_ip, corpus_line))
    {
        std::vector < std::string > words;
        boost::split(words, corpus_line, boost::is_any_of(" "));
        libsvmcorpus_op << temp_i << "\t";
        for (auto word :words) {
            auto word_id = std::distance(vocab_map.begin(),vocab_map.find(word));
            if (std::find(unique_words_inline.begin(), unique_words_inline.end(), word) == unique_words_inline.end())
            {
                unique_words_inline.push_back(word);
                auto word_count = std::count (words.begin(), words.end(), word);
                libsvmcorpus_op << word_id << ':' << word_count << ' ' ;
            }
        }
        libsvmcorpus_op << std::endl;
        unique_words_inline.clear();
        temp_i++;
        }
    }

void LightLDAWrapper::generate_binary_from_libsvm(std::string output_data_path, std::string lda_path) {
    std::cout << "Generate the binary inputs from the libsvm format inputs" << std::endl;
    std::string libsvm_vocab_path = output_data_path + "libsvm_vocab.dat";
    std::string libsvm_corpus_path = output_data_path + "libsvm_corpus.dat";
    std::cout << output_data_path << std::endl;
    auto convert_to_bin = "./dump_binary " + libsvm_corpus_path + " " + libsvm_vocab_path + " " + output_data_path + " 0";
    char *convertbin = &convert_to_bin[0u];
    chdir(lda_path.c_str());
    system(convertbin);
}


void LightLDAWrapper::apply_lda_on_binary(std::string output_data_path, std::string lda_path) {
    std::cout << "Aplly LightLDA on binary inputs" << std::endl;

    std::string apply_lda = "./lightlda -num_vocabs 9000 -num_topics 50 -num_iterations 50 -alpha 0.01 "
                                    "-beta 0.1 -max_num_document 5000 -input_dir " + output_data_path;
    chdir(lda_path.c_str());
    system(apply_lda.c_str());
    std::string from_1 = lda_path + "doc_topic.0";
    std::string from_2 = lda_path + "server_0_table_0.model";
    std::string from_3 = lda_path + "server_0_table_1.model";
    std::string to_1 = output_data_path + "doc_topic.0";
    std::string to_2 = output_data_path + "server_0_table_0.model";
    std::string to_3 = output_data_path + "server_0_table_1.model";
    boost::filesystem::copy_file(from_1, to_1);
    boost::filesystem::copy_file(from_2, to_2);
    boost::filesystem::copy_file(from_3, to_3);
}


void LightLDAWrapper::get_gamma_lambda(std::string lda_path) {
    std::cout << "Get gamma and lambda from the output of lda" << std::endl;

}