```mermaid
flowchart LR
    U["User browser"] --> S["Streamlit app: UI and backend"]

    subgraph D ["Market data layer"]
        Y["yfinance (Yahoo Finance)"]
        Q["Stooq (pandas_datareader)"]
        A["Alpha Vantage (optional, API key)"]
        C["Custom CSV upload"]
    end

    S --> D
    D --> FE["Feature engineering"]
    FE --> SC["StandardScaler (scikit-learn)"]
    SC --> HMM["GaussianHMM (hmmlearn)"]

    HMM --> LBL["Regime labelling"]
    LBL --> STATS["Regime statistics (return, vol, duration)"]
    STATS --> VIZ["Plotly charts (price, regimes, confidence)"]
    VIZ --> S

    %% Optional / inactive features
    subgraph OPT ["Optional features (inactive)"]
        PAY["st_paywall (Google OAuth + Stripe)"]
    end
    S -. future auth .- PAY

    subgraph PREMIUM ["Premium module (future)"]
        PORT["Portfolio regime analytics"]
    end
    HMM --> PORT
    PORT --> VIZ
