function WelcomePage({ user }) {
    const history = useHistory();
  
    if (!user) {
      return <div>Loading...</div>;
    }
  
    const handleClose = () => {
      history.push('/'); // 메인 페이지 또는 로그인 페이지로 이동
    };
  
    return (
      // 기존 코드...
      <button
        onClick={handleClose}
        className="w-full bg-indigo-600 text-white p-2 rounded hover:bg-indigo-700"
      >
        닫기
      </button>
    );
  }
  