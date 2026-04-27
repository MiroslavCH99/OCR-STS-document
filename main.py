import argparse

from loguru import logger

from config import load_rules
from service import process_folder


def parse_args() -> argparse.Namespace:
    """Разбирает аргументы командной строки."""
    parser = argparse.ArgumentParser(
        description="Извлечение ФИО собственника из изображений СТС и сохранение результата в CSV."
    )

    parser.add_argument(
        "--input_folder",
        default="data",
        help="Папка с изображениями. По умолчанию: data",
    )

    parser.add_argument(
        "--output_csv",
        default="fio_result.csv",
        help="Путь к CSV-файлу с результатами. По умолчанию: fio_result.csv",
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Искать изображения также во вложенных папках.",
    )

    parser.add_argument(
        "--debug_candidates",
        action="store_true",
        help="Выводить в логи детальную информацию по кандидатам ФИО.",
    )

    return parser.parse_args()


def main() -> None:
    """Точка входа: загрузка настроек, запуск пакетной обработки."""
    args = parse_args()

    logger.info("Запуск скрипта")
    logger.info("Аргументы запуска: {}", vars(args))

    rules = load_rules()

    total = process_folder(
        input_folder=args.input_folder,
        output_csv=args.output_csv,
        rules=rules,
        recursive=args.recursive,
        debug_candidates=args.debug_candidates,
    )

    logger.success("Обработано файлов: {}", total)
    logger.success("CSV сохранен: {}", args.output_csv)


if __name__ == "__main__":
    main()
