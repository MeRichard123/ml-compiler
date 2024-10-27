#define RICHBUILD_IMPLEMENTATION
#include "richBuild.h"

#define cflags "-Wall"
#define executable_name "main"

// ============= Run Lua Examples ==================

void BUILD_LUA(char* program) {
  char cmd[256];
  const char* compiler = "../Lua/lua-5.4.7/src/lua";
  
  // append the compile command
  snprintf(cmd, sizeof(cmd), "%s ./Lua/%s.lua", compiler, program);

  INFO("Compiling Command:");
  CMD(cmd);
  
  system(cmd);
}

// ============= Generate Example Files ===============

void write_to_example_file(char* program_file, char* output, char* program_name)
{
  /*
    * This function takes the neccessary data and write it into a new file
  */
  // pointers to the program and new example file
  FILE* program_file_ptr;
  FILE* test_file_ptr;

  // reading the program
  program_file_ptr = fopen(program_file, "r");
  
  // format the file name
  char example_file_name[256];
  str_remove(&program_name, ".lua");
  snprintf(example_file_name, sizeof(example_file_name) ,"./training_examples/%s.luax", program_name);
  
  // Create a new file for the example
  test_file_ptr = fopen(example_file_name, "a");
  
  if (program_file_ptr == NULL) 
  {
    fprintf(stderr, "Unable to open file");
  }

  if (test_file_ptr == NULL) 
  {
    fprintf(stderr, "Couldn't create file");
  }

  // Read the files character by character and write to the file
  char ch; 
  while ((ch = fgetc(program_file_ptr)) != EOF)
  {
    fputc(ch, test_file_ptr);
  }
  fprintf(test_file_ptr, "\n<PROGRAM END>\n\n");
  fprintf(test_file_ptr, output);
  fclose(program_file_ptr);
}


const size_t NUMBER_OF_EXAMPLES = 10;

char* example_files[NUMBER_OF_EXAMPLES];
char* program_names[NUMBER_OF_EXAMPLES];


void BUILD_EXAMPLES()
{
  /*
  * Read all the Lua files, and compute their outputs. 
  * Then save to the examples. 
  */ 
  const char* compiler = "../Lua/lua-5.4.7/src/lua";
  // Take all the Lua Files and store the names and directories
  DIR *directory;
  struct dirent *dir;

  directory = opendir("./Lua");
   
  if (directory) {
    int counter = 0;
    while((dir = readdir(directory)) != NULL) {
      if (strcmp(dir->d_name, "..") > 0) {
        example_files[counter] = malloc(256);
        snprintf(example_files[counter], 256, "./Lua/%s", dir->d_name);
        program_names[counter] = dir->d_name;
        counter++;
      }
    }
    closedir(directory);
  }
  // compute the outputs for each file
  for (int i = 0; i < NUMBER_OF_EXAMPLES; i++)
  {
    if (example_files[i] != NULL) {
      // create the running command
      char cmd[256];
      snprintf(cmd, sizeof(cmd), "%s %s", compiler, example_files[i]);
      // run and store the stdout buffer stream
      FILE *stream = popen(cmd,"r");
      
      char buffer[1024];
      char *line_p = fgets(buffer, sizeof(buffer), stream);
      pclose(stream);
      // write to the example file
      if (line_p != NULL) {
        write_to_example_file(example_files[i], line_p, program_names[i]);
      }
    }
  }
}


int main(int argc, char** argv) {
  if (argc == 3 && strcmp(argv[1],"--lua") == 0) 
  {
    BUILD_LUA(argv[2]);
  }
  else if (argc == 2 && strcmp(argv[1],"--lua") == 0) 
  {
    WARN("Input Error: No Input File Provided");
  }
  else if (argc == 2 && strcmp(argv[1], "--examples") == 0) {
    INFO("Building Examples...");
    BUILD_EXAMPLES();
    INFO("Finished Building Examples: './examples/'");
  }
  return 0;
}
