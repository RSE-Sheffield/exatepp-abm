#pragma once

/**
 * the model's "main" method, which is called by main.cu::main. This split allows testing of this method.
 *
 * @param argc argc forwarded from the main method
 * @param argv argv forwarded from the main method
 * @return exit code
 */
int entrypoint(int argc, const char **argv);