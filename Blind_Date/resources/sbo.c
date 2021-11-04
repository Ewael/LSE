// includes...

void vuln(void)
{
    char buffer[32]; // not initialized
    read(0, buffer, INPUT_SIZE); // we do not know how many bytes it reads
    printf("Welcome to the LSE, %s\n", buffer); // safe printf
    return; // <-- vulnerable return
}

int main(void)
{
    printf("Hello you!\nWhat's your name?\n>>> ");
    vuln();
    printf("Bye!\n");
    return 0;
}
