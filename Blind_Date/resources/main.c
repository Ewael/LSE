// includes...

int main(void)
{
    char username[SIZE]; // we do not know SIZE yet
    // [...] <- get input with `scanf` or `gets` or whatever
    printf("Welcome to the LSE, ");
    printf(username); // <--- unsafe line
    printf("\nBye!\n");
    return 0;
}
