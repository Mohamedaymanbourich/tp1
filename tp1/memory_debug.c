#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SIZE 5

int* allocate_array(int size) {
	int *arr = (int*)malloc(size * sizeof(int));
	if (!arr) {
		fprintf(stderr, "Memory allocation failed\n");
		exit(EXIT_FAILURE);
	}
	return arr;
}

void initialize_array(int *arr, int size) {
	if (!arr) return;
	for (int i = 0; i < size; i++) {
		arr[i] = i * 10;
	}
}

void print_array(int *arr, int size) {
	if (!arr) return;
	printf("Array elements: ");
	for (int i = 0; i < size; i++) {
		printf("%d ", arr[i]);
	}
	printf("\n");
}

int* duplicate_array(int *arr, int size) {
	if (!arr) return NULL;
	int *copy = (int*)malloc(size * sizeof(int));
	if (!copy) {
		fprintf(stderr, "Memory allocation failed\n");
		exit(EXIT_FAILURE);
	}
	memcpy(copy, arr, size * sizeof(int));
	return copy;
}

void free_memory(int *arr) {
	// add free memory line to fix the memory leak
	free(arr);
}

int main() {
	int *array = allocate_array(SIZE);
	initialize_array(array, SIZE);
	print_array(array, SIZE);
	int *array_copy = duplicate_array(array, SIZE);
	print_array(array_copy, SIZE);
	free_memory(array);
    free_memory(array_copy);
	return 0; // Memory leak on purpose
}