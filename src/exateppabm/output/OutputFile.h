#pragma once

#include <filesystem>
#include <string>

namespace exateppabm {
namespace output {


class OutputFile {

 public:
    /**
     * Constructor
     */
    OutputFile(std::filesystem::path path) : _filepath(path), _handle(nullptr) { }

    /**
     * Dtor
     */
    ~OutputFile() { this->close(); }

    /**
     * Get the directory in which the file will be created
     */
    std::filesystem::path getDirectory() { return this->_filepath.parent_path(); };

    /**
     * Get the path to the file;
     */
    std::filesystem::path getFilepath() { return this->_filepath; };

    /**
     * Open the file handle for writing
     */
    void open() {
        // @todo - validate the file is not a directory first? and make any parent dirs?
        this->_handle = std::fopen(this->getFilepath().c_str(), "w");
    }

    /**
     * Close the file handle
     */
    void close(){
        if(this->_handle) {
            std::fclose(this->_handle);
            this->_handle = nullptr;
        }
    }

    /**
     * Pure Virtual function for writing out data to the file.
     */
    virtual bool write() = 0;

 protected:

    /**
     * Path to the file's intended output location
     */
    std::filesystem::path _filepath;
    
    /**
     * handle to the csv file (if exists)
     */
    std::FILE* _handle;

};

}  // namespace exateppabm
}  // namespace output
