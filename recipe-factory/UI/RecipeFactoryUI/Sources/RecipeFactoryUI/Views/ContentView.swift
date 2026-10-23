import SwiftUI

struct ContentView: View {
    var body: some View {
        TabView {
            DashboardView()
                .tabItem { Label("Dashboard", systemImage: "gauge.with.dots.needle.50percent") }
            RecipeListView()
                .tabItem { Label("Recipes", systemImage: "list.bullet.rectangle") }
            ProfilePickerView()
                .tabItem { Label("Profiles", systemImage: "slider.horizontal.3") }
            TradeoffExplorerView()
                .tabItem { Label("Trade-offs", systemImage: "arrow.left.arrow.right") }
            LedgerView()
                .tabItem { Label("Ledger", systemImage: "books.vertical") }
        }
        .padding()
    }
}
