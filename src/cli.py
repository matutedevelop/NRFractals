import argparse
from lib import ComplexPolynomial, mutate_df, nr_data, plot_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--coefs", type=float, nargs="+")

    arguments = parser.parse_args()

    if arguments.coefs is not None:
        poly = ComplexPolynomial(arguments.coefs)
    else:
        while True:
            try:
                degree = int(input("Enter the degree of the desired polynomial:  "))
            except Exception as e:  # noqa: F841
                print("degree should be an integer")
            else:
                break
        coefs = []
        for i in range(degree + 1):
            try:
                coef = float(input(f"Enter the coeficient of the x^{i} term:  "))
            except Exception as e:  # noqa: F841
                print("coeficient  should be an real number")

            coefs.append(coef)

        poly = ComplexPolynomial(coefs)


    print(poly)

    df = nr_data(poly)
        

    df = mutate_df(df, poly.deg + 1)


    plot_image(df)
