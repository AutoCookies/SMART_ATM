import React, { useState, useEffect } from 'react';
import axios from 'axios';

const UserManagement = () => {
  const [users, setUsers] = useState([]);

  // Sử dụng useEffect để gửi yêu cầu tới backend khi component được render
  useEffect(() => {
    try {
      const fetch = async () => {
        const data = await axios.get("http://localhost:5000/api/users");
        setUsers(data.data);
      };
      fetch();
    } catch (error) {
      console.log(error);
    }
  }, []);  // Tham số mảng rỗng [] có nghĩa là useEffect sẽ chỉ chạy một lần khi component được render
  // console.log(users)
  // Hiển thị ngày login gần nhất từ mảng loginLog
  const formatDate = (dateString) => {
    const date = new Date(dateString);

    const day = String(date.getUTCDate()).padStart(2, '0');
    const month = String(date.getUTCMonth() + 1).padStart(2, '0'); // Tháng trong JavaScript bắt đầu từ 0
    const year = date.getUTCFullYear();
    const hours = String(date.getUTCHours()).padStart(2, '0');
    const minutes = String(date.getUTCMinutes()).padStart(2, '0');

    return `${day}-${month}-${year} ${hours}:${minutes}`;
  };
  return (
    <div className="container mx-auto p-6">
      <h1 className="text-3xl font-bold mb-6 text-center">User Management</h1>

      <div className="shadow overflow-hidden border-b border-gray-200 sm:rounded-lg">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Username</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">History Login</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Created At</th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {users.map(user => (
              <tr key={user._id}>
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="text-sm font-medium text-gray-900">{user.username}</div>
                </td>
                <td className="flex flex-col justify-start px-6 py-4 whitespace-nowrap">
                  {
                    user.loginLog.map(item => (
                      <span className="text-sm text-gray-500">{item?.loginTime}</span>

                    ))
                  }

                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="text-sm text-gray-500">{new Date(user?.createdAt).toLocaleString()}</div> {/* Hiển thị ngày tạo */}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default UserManagement;
