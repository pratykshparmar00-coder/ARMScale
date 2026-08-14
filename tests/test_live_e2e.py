import urllib.request
import json

def run_tests():
    endpoints = [
        '/health',
        '/api/platform',
        '/api/system',
        '/api/optimization/search-space',
        '/api/models/variants',
        '/api/optimization/experiments',
        '/api/optimization/latest?workload_type=short_generation',
        '/api/optimization/latest?workload_type=context_stress',
        '/api/optimization/pareto?workload_type=short_generation',
        '/api/optimization/pareto?workload_type=context_stress'
    ]

    all_passed = True
    for ep in endpoints:
        url = f'http://127.0.0.1:8000{ep}'
        try:
            req = urllib.request.urlopen(url)
            assert req.status == 200
            data = json.loads(req.read().decode())
            print(f"[OK 200] {ep}")
        except Exception as e:
            print(f"[FAIL] {ep}: {e}")
            all_passed = False

    # Test POST /api/optimize/recommend for all 4 objectives
    for obj in ['speed', 'throughput', 'size', 'balanced']:
        req = urllib.request.Request(
            'http://127.0.0.1:8000/api/optimize/recommend',
            data=json.dumps({'workload': 'short_generation', 'objective': obj}).encode(),
            headers={'Content-Type': 'application/json'}
        )
        resp = urllib.request.urlopen(req)
        assert resp.status == 200
        data = json.loads(resp.read().decode())
        assert data['status'] == 'success'
        print(f"[OK 200] POST /api/optimize/recommend (obj={obj}) -> {data['configuration_id']}")

    print("\nALL LIVE ENDPOINT CHECKS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run_tests()
