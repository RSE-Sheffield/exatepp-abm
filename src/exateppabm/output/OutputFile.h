#pragma once

#include <cstdio>
#include <filesystem>
#include <string>

namespace exateppabm {
namespace output {

/**
 * Class representing an OutputFile, with generic methods commonly used for output files
 */
class OutputFile {
 public:
    /**
     * Constructor
     * @param path path to location for output file
     */
    explicit OutputFile(std::filesystem::path path) : _filepath(path), _handle(nullptr) { }

    /**
     * Dtor
     */
    ~OutputFile() { this->close(); }

    /**
     * Get the directory in which the file will be created
     * @return path to directory file will be written to
     */
    std::filesystem::path getDirectory() { return this->_filepath.parent_path(); }

    /**
     * Get the path to the file
     * @return output file path
     */
    std::filesystem::path getFilepath() { return this->_filepath; }

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
    void close() {
        if (this->_handle) {
            std::fclose(this->_handle);
            this->_handle = nullptr;
        }
    }

    /**
     * Function for writing the body of the output file
     */
    virtual bool write() = 0;

 protected:
    /**
     * Path to the file's intended output location
     */
    std::filesystem::path _filepath;

    /**
     * Handle to the csv file (if exists)
     */
    std::FILE* _handle;
};

}  // namespace output
}  // namespace exateppabm
