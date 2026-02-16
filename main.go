
package main
import (
	"bufio"
	"context"
	"fmt"
	"log"
	"os"
	"strings"
	"github.com/sashabaranov/go-openai"
)

func main() {
	// Чтение API ключа из переменной окружения
	apiKey := os.Getenv("ROUTERAPI_API_KEY")
	if apiKey == "" {
		log.Fatal("ROUTERAPI_API_KEY environment variable not set")
	}
	baseURL := "https://routerai.ru/api/v1"

	// Конфигурация клиента с кастомным baseURL
	config := openai.DefaultConfig(apiKey)
	config.BaseURL = baseURL

	// Создание клиента
	client := openai.NewClientWithConfig(config)

	// Запрос ввода сообщения от пользователя
	fmt.Print("Введите ваш запрос: ")
	reader := bufio.NewReader(os.Stdin)
	userMessage, err := reader.ReadString('\n')
	if err != nil {
		log.Fatalf("Ошибка при чтении ввода: %v", err)
	}
	userMessage = strings.TrimSpace(userMessage)

	if userMessage == "" {
		log.Fatal("Сообщение не может быть пустым")
	}

	// Формирование запроса
	resp, err := client.CreateChatCompletion(
		context.Background(),
		openai.ChatCompletionRequest{
			Model: "deepseek/deepseek-v3.2",
			Messages: []openai.ChatCompletionMessage{
				{
					Role:    openai.ChatMessageRoleUser,
					Content: userMessage,
				},
			},
		},
	)

	if err != nil {
		log.Fatalf("Ошибка при вызове ChatCompletion: %v", err)
	}

	// Вывод ответа
	if len(resp.Choices) > 0 {
		fmt.Println("Ответ:", resp.Choices[0].Message.Content)
	} else {
		fmt.Println("Пустой ответ от API")
	}
}
