#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

// gcc -o chall chall.c -no-pie
// gcc gcc -o chall chall.c -no-pie -Xlinker -rpath=./ -Xlinker -I./ld-2.27.so

void vuln(void)
{
    char name[32];
    puts("What's your name?");
    printf(">>> ");
    fflush(stdout);
    read(0, name, 0x80);
    printf("Welcome to the LSE, %s", name);
}

int main(void)
{
    puts("Hello you!");
    vuln();
    puts("Bye!");
    return 0;
}
