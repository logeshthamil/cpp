//
// Created by lt on 6/2/17.
//

#include <iostream>
#include <fstream>
#include <map>
#include "LightLDAWrapper.h"
#include <boost/algorithm/string.hpp>
#include <sstream>
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
    for (const auto &p : mp)
    {
//        std::cout << p->first << "\t" << std::to_string(p->second) << std::endl;
        of << p.first << std::endl;
    }

}

double sum_vector(std::vector<int> const &v) {
    return 1.0 * std::accumulate(v.begin(), v.end(), 0LL);
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
//    std::cout << output_data_path << std::endl;
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
    if (boost::filesystem::exists(to_1))
        boost::filesystem::remove(to_1);
    if (boost::filesystem::exists(to_2))
        boost::filesystem::remove(to_2);
    if (boost::filesystem::exists(to_3))
        boost::filesystem::remove(to_3);
    boost::filesystem::copy_file(from_1, to_1);
    boost::filesystem::copy_file(from_2, to_2);
    boost::filesystem::copy_file(from_3, to_3);
}


void LightLDAWrapper::get_gamma_lambda(std::string output_data_path) {
    std::cout << "Get gamma and lambda from the output of lda" << std::endl;
    std::string gamma_path = output_data_path + "gamma.dat";
    std::string lambda_path = output_data_path + "lambda.dat";
    std::string doc_topic_hashed_path = output_data_path + "doc_topic.0";
    std::string topic_word_hashed_path = output_data_path + "server_0_table_0.model";
    std::string total_topic_hashed_path = output_data_path + "server_0_table_1.model";
    std::fstream doc_topic_ip(doc_topic_hashed_path);
    std::ofstream doc_topic_out(gamma_path);
    std::fstream topic_word_ip(topic_word_hashed_path);
    std::ofstream topic_word_out(lambda_path);
    std::fstream total_topic_ip(total_topic_hashed_path);
    int num_topics = 0;
    std::string total_topic_line;
    while (getline(total_topic_ip, total_topic_line)) {
        std::vector<std::string> each_topic;
        boost::split(each_topic, total_topic_line, boost::is_any_of(" "));
        num_topics = each_topic.end() - each_topic.begin() - 1;
    }
//    std::cout << num_topics << std::endl;

//  write the gamma matrix to a file from the doc topic hashed input
    std::string doc_topic_line;
    while (std::getline(doc_topic_ip, doc_topic_line)) {
        std::vector<std::string> words;
        boost::split(words, doc_topic_line, boost::is_any_of(" "));
        words.erase(words.begin());
        std::vector<int> topics;
        std::vector<int> topicproportions;
        for (auto word :words) {
            if (!word.empty()) {
                std::vector<std::string> topic_prop;
                boost::split(topic_prop, word, boost::is_any_of(":"));
                int t_prop;
                int t;
                std::istringstream buffer1(topic_prop.front());
                buffer1 >> t;
                std::istringstream buffer2(topic_prop.back());
                buffer2 >> t_prop;
                topics.push_back(t);
                topicproportions.push_back(t_prop);
            }
        }
        double sum = sum_vector(topicproportions);
        std::vector<double> doc_topic_eachline(num_topics, 0.0);
        int i = 0;
        for (int t : topics) {
//            std::cout << t << " " << topicproportions.at(i)/sum << std::endl;
            doc_topic_eachline[t] = topicproportions.at(i) / sum;
            i++;
        }
        for (std::vector<double>::const_iterator i = doc_topic_eachline.begin(); i != doc_topic_eachline.end(); ++i) {
            doc_topic_out << *i << " ";
        }
        doc_topic_out << "\n";
    }
    doc_topic_ip.close();
    doc_topic_out.close();

//      write the gamma matrix to a file from the doc topic hashed input
    std::string topic_word_line;
    while (std::getline(topic_word_ip, topic_word_line)) {
        std::vector<std::string> words;
        boost::split(words, topic_word_line, boost::is_any_of(" "));
        words.erase(words.begin());
        std::vector<int> topics;
        std::vector<int> topicproportions;
        for (auto word :words) {
            if (!word.empty()) {
                std::vector<std::string> topic_prop;
                boost::split(topic_prop, word, boost::is_any_of(":"));
                int t_prop;
                int t;
                std::istringstream buffer1(topic_prop.front());
                buffer1 >> t;
                std::istringstream buffer2(topic_prop.back());
                buffer2 >> t_prop;
                topics.push_back(t);
                topicproportions.push_back(t_prop);
            }
        }
        double sum = sum_vector(topicproportions);
        std::vector<double> topic_word_eachline(num_topics, 0.0);
        int i = 0;
        for (int t : topics) {
//            std::cout << t << " " << topicproportions.at(i)/sum << std::endl;
            topic_word_eachline[t] = topicproportions.at(i) / sum;
            i++;
        }
        for (std::vector<double>::const_iterator i = topic_word_eachline.begin(); i != topic_word_eachline.end(); ++i) {
            topic_word_out << *i << " ";
        }
        topic_word_out << "\n";
    }
    topic_word_ip.close();
    topic_word_out.close();

}
