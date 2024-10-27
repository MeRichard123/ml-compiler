#ifndef RICHBUILD_H
#define RICHBUILD_H

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <dirent.h>
#include <stdarg.h>

// Common Packages

#define RAYLIB "-lraylib `pkg-config --libs --cflags raylib`"

// Declarations

// Find if a string ends with a specfied suffix 
// Returns 1 if true and 0 if false
int str_ends_with(const char* string, const char* suffix);

// Find if a string starts with a specifed prefic
// Returns 1 if true and 0 if fase 
int str_starts_with(const char* string, const char* prefix); 

// Remove a substring (sub) from a string (str)
void str_remove(char** str, char* sub);

// count the number of spaces in a string 
int count_files(const char* string);

// return a string containing all the C files in the current 
// directory excluding the build script itself
char* get_files();

// remove files from the build
void exclude_files(char** files, int size, ...);

// add everything to the compile command and run it
void compile_files(char* compiler, const char* files, const char* cflags, char* executable_name, char* packages);

// Macro Definitions

#define READ_FILES() get_files()
#define CLEANUP() clean_up()

#define COMPILE(compiler, files, cflags, executable_name, packages) \
  compile_files(compiler, files, cflags, executable_name, packages)

#define EXCLUDE(files, ...) \
  exclude_files(files, count_files(*files), __VA_ARGS__)

#define INFO(msg) printf("[INFO] %s\n", msg)
#define WARN(msg) printf("[WARN] %s\n", msg)
#define CMD(cmd) printf("[CMD] %s\n", cmd)
#define DONE(msg, file) printf("[DONE] %s : %s\n", msg, file)

#endif // RICHBUILD_H


#ifdef RICHBUILD_IMPLEMENTATION

// top level constants
const size_t NUMBER_OF_FILES_TO_BUILD = 40;
const size_t FILENAME_SIZE_LIMIT = 256;

char* file_list;

char* get_files() {
  DIR *directory;
  struct dirent *dir;

  // allocate memory for the file names
  file_list = (char*)malloc(sizeof(char) * FILENAME_SIZE_LIMIT * NUMBER_OF_FILES_TO_BUILD);

  directory = opendir("./src");
  if (directory) {
    strcpy(file_list, "");

    while ((dir = readdir(directory)) != NULL) {
      if (str_ends_with(dir->d_name, ".c") && !str_starts_with(dir->d_name, "richBuild")) {
        strcat(file_list, "./src/");
        strcat(file_list, dir->d_name);
        strcat(file_list, " ");
      }
    }
    closedir(directory);
  }
  return file_list;
}

int count_files(const char* string) {
  int count = 0;
  const char* s;

  for (s = string; *s != '\0'; s++) {
    if (*s == ' ') {
      count++;
    }
  }

  return count;
}

void str_remove(char** str, char* sub) {
  size_t len = strlen(sub);
  
  if (len > 0) {
    char* p = *str;
    while ((p = strstr(p, sub)) != NULL) {
      memmove(p, p + len, strlen(p + len) + 1);
    }
    str = &p;
  }
}


int str_ends_with(const char* string, const char* suffix) {
  if (!string || !suffix) {
    return 0;
  }
  
  size_t string_len = strlen(string);
  size_t suffix_len = strlen(suffix);

  if (suffix_len > string_len) {
    return 0;
  }

  return strncmp(string + string_len - suffix_len, suffix, suffix_len) == 0;
}

int str_starts_with(const char* string, const char* prefix) {
  if (!prefix || !string) {
    return 0;
  }

  size_t prefix_length = strlen(prefix);

  if (prefix_length > strlen(string)) {
    return 0;
  }

  return strncmp(string, prefix, prefix_length) == 0;
}

void exclude_files(char** files, int size, ...) {
  va_list list_ptr;
  va_start(list_ptr, size);

  for (int i = 0; i < size - 1; i++) {
    char* word = va_arg(list_ptr, char *);
    str_remove(files, word);
  }

  va_end(list_ptr);
}

void compile_files(char* compiler, const char* files, const char* cflags, char* executable_name, char* packages) {
  char compile_command[256];

  if (packages == NULL) {
    snprintf(compile_command, sizeof(compile_command),
        "%s %s -o %s %s", 
        compiler, cflags, executable_name, files);
  } else {
    snprintf(compile_command, sizeof(compile_command),
        "%s %s -o %s %s %s", 
        compiler, cflags, executable_name, files, packages);
  }

  printf("%s", compile_command);

  INFO("Compiling Command:");
  CMD(compile_command);
  system(compile_command);
  DONE("Finshed Compiling", executable_name);
}

void clean_up() {
  free(file_list);
}


#endif // RICHBUILD_IMPLEMENTATION
